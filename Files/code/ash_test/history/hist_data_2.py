"""
DROP-IN REPLACEMENT (adds human-readable dates)

Adds:
- human-readable columns:
    - gdelt_datetime (ISO 8601 from gdelt_seendate)
    - publish_datetime (ISO 8601 from publish_date when parseable)
    - event_datetime (the best “usable” datetime = publish_datetime if present else gdelt_datetime)
- keeps your existing columns unchanged for compatibility

Also retains:
- improved publish_date extraction (meta + JSON-LD + normalization)
- 2021-01 through 2025-12 loop
"""

import os
import re
import time
import json
import random
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from newspaper import Article

# Optional extractors
try:
    import trafilatura  # pip install trafilatura
except Exception:
    trafilatura = None

try:
    from readability import Document  # pip install readability-lxml
except Exception:
    Document = None

try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
except Exception:
    BeautifulSoup = None

try:
    from dateutil import parser as dateparser  # pip install python-dateutil
except Exception:
    dateparser = None


# -----------------------------
# Config
# -----------------------------

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

@dataclass
class ScrapeConfig:
    data_dir: str
    keywords: str
    domains: List[str]
    year: int
    month: int

    max_workers: int = 10
    max_per_domain: int = 2
    gdelt_chunk_days: int = 7
    gdelt_maxrecords: int = 250

    gdelt_sleep_base: float = 1.5
    domain_sleep_base: float = 2.0

    timeout_connect: float = 5.0
    timeout_read: float = 25.0

    min_content_chars: int = 200

    output_format: str = "csv"  # "csv" or "parquet" (parquet requires pyarrow)


# -----------------------------
# Logging
# -----------------------------

def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("gdelt_news_scraper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# -----------------------------
# HTTP session w/ retries
# -----------------------------

def build_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )
    return session


# -----------------------------
# URL canonicalization
# -----------------------------

_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid",
    "ref", "ref_src", "spm", "cmpid"
}

def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        fragmentless = p._replace(fragment="")
        q = parse_qsl(fragmentless.query, keep_blank_values=True)
        q2 = [(k, v) for (k, v) in q if k.lower() not in _TRACKING_PARAMS]
        q2.sort(key=lambda x: (x[0].lower(), x[1]))
        new_query = urlencode(q2, doseq=True)

        path = fragmentless.path or "/"
        path = re.sub(r"/amp/?$", "/", path, flags=re.IGNORECASE)

        cleaned = fragmentless._replace(path=path, query=new_query)
        netloc = cleaned.netloc.replace(":80", "").replace(":443", "")
        cleaned = cleaned._replace(netloc=netloc)
        return urlunparse(cleaned)
    except Exception:
        return url


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# -----------------------------
# Date helpers (NEW)
# -----------------------------

def _parse_any_datetime(value: str) -> Optional[datetime]:
    """
    Parse a datetime string into a datetime object (best-effort).
    Supports:
    - GDELT seendate formats like '20210215T010000Z' and '2021-02-15 01:00:00'
    - ISO / RFC formats via dateutil (if installed)
    """
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None

    # Common GDELT form: YYYYMMDDTHHMMSSZ
    try:
        if re.fullmatch(r"\d{8}T\d{6}Z", s):
            dt = datetime.strptime(s, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            return dt
    except Exception:
        pass

    # Some GDELT outputs: YYYY-MM-DD HH:MM:SS
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", s):
            # unknown tz; treat as naive
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

    # ISO-like
    if dateparser is not None:
        try:
            return dateparser.parse(s)
        except Exception:
            return None

    return None


def _to_human(dt: Optional[datetime]) -> str:
    """Human-readable string; keeps timezone if present."""
    if dt is None:
        return ""
    # Example: '2021-02-15 01:00:00 UTC' or without tz
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# -----------------------------
# Publish date parsing (best-effort, improved)
# -----------------------------

def _normalize_date(value: str) -> Optional[str]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None

    dt = _parse_any_datetime(value)
    if dt:
        # normalize to ISO (keep tz if present)
        return dt.isoformat()

    return None


def extract_publish_date_from_html(html: str) -> Optional[str]:
    if not html or BeautifulSoup is None:
        return None

    try:
        soup = BeautifulSoup(html, "html.parser")

        meta_keys = [
            ("property", "article:published_time"),
            ("property", "og:published_time"),
            ("name", "article:published_time"),
            ("name", "pubdate"),
            ("name", "publishdate"),
            ("name", "publish_date"),
            ("name", "date"),
            ("name", "dc.date"),
            ("name", "DC.date.issued"),
            ("name", "sailthru.date"),
            ("itemprop", "datePublished"),
        ]
        for attr, key in meta_keys:
            tag = soup.find("meta", attrs={attr: key})
            if tag and tag.get("content"):
                norm = _normalize_date(tag["content"])
                if norm:
                    return norm

        for script in soup.find_all("script", type="application/ld+json"):
            raw = (script.string or script.get_text() or "").strip()
            if not raw:
                continue

            try:
                data = json.loads(raw)
            except Exception:
                continue

            objs = data if isinstance(data, list) else [data]
            expanded: List[dict] = []
            for obj in objs:
                if isinstance(obj, dict) and "@graph" in obj and isinstance(obj["@graph"], list):
                    expanded.extend([x for x in obj["@graph"] if isinstance(x, dict)])
            objs = objs + expanded

            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                typ = obj.get("@type") or obj.get("type")
                if isinstance(typ, list):
                    typ = " ".join(map(str, typ))
                if typ and any(t in str(typ) for t in ["NewsArticle", "Article", "BlogPosting", "Reportage"]):
                    dp = obj.get("datePublished") or obj.get("dateCreated") or obj.get("dateModified")
                    if isinstance(dp, str):
                        norm = _normalize_date(dp)
                        if norm:
                            return norm

    except Exception:
        return None

    return None


# -----------------------------
# GDELT fetch
# -----------------------------

def fetch_gdelt_single_domain(
    session: requests.Session,
    logger: logging.Logger,
    keywords: str,
    domain: str,
    start_date: datetime,
    end_date: datetime,
    config: ScrapeConfig,
    timeout: Tuple[float, float] = (5.0, 25.0),
) -> List[dict]:
    all_data: List[dict] = []
    current = start_date
    query = f'{keywords} domain:{domain} sourcelang:eng'

    while current < end_date:
        chunk_end = min(current + timedelta(days=config.gdelt_chunk_days), end_date)

        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": config.gdelt_maxrecords,
            "format": "json",
            "startdatetime": current.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
        }

        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                data = resp.json()
                articles = data.get("articles", []) or []
                all_data.extend(articles)
            else:
                logger.info(
                    f"GDELT non-OK domain={domain} {current.date()}..{chunk_end.date()} status={resp.status_code}"
                )
        except Exception as e:
            logger.warning(
                f"GDELT error domain={domain} {current.date()}..{chunk_end.date()} err={type(e).__name__}: {e}"
            )

        current = chunk_end
        time.sleep(config.gdelt_sleep_base + random.random() * 1.2)

    return all_data


def fetch_gdelt_urls_multi_domain(
    session: requests.Session,
    logger: logging.Logger,
    keywords: str,
    domains: List[str],
    start_date: datetime,
    end_date: datetime,
    config: ScrapeConfig,
) -> pd.DataFrame:
    all_articles: List[dict] = []

    gdelt_min = datetime(2017, 2, 19)
    if start_date < gdelt_min:
        start_date = gdelt_min

    logger.info(f"Searching {len(domains)} domains separately")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    for idx, domain in enumerate(domains, 1):
        logger.info(f"[{idx}/{len(domains)}] Fetching URLs for {domain}")

        articles = fetch_gdelt_single_domain(
            session=session,
            logger=logger,
            keywords=keywords,
            domain=domain,
            start_date=start_date,
            end_date=end_date,
            config=config,
            timeout=(config.timeout_connect, config.timeout_read),
        )
        logger.info(f"  -> {len(articles)} URLs (raw)")
        all_articles.extend(articles)

        if idx < len(domains):
            time.sleep(config.domain_sleep_base + random.random() * 2.0)

    df = pd.DataFrame(all_articles)
    if df.empty:
        return df

    if "url" in df.columns:
        df["canonical_url"] = df["url"].astype(str).map(canonicalize_url)

    keep_cols = [c for c in ["seendate", "title", "domain", "url", "canonical_url"] if c in df.columns]
    df = df[keep_cols].copy()

    if "canonical_url" in df.columns:
        df = df.drop_duplicates(subset=["canonical_url"], keep="first")

    if "title" in df.columns and "seendate" in df.columns:
        def seen_day(x):
            try:
                return str(x)[:10]
            except Exception:
                return ""
        df["seen_day"] = df["seendate"].map(seen_day)
        df["title_key"] = (
            df["title"]
            .fillna("")
            .str.strip()
            .str.lower()
            .map(lambda t: re.sub(r"\s+", " ", t))
        )
        df = df.drop_duplicates(subset=["title_key", "seen_day"], keep="first")

    return df.reset_index(drop=True)


# -----------------------------
# Extraction (tiered)
# -----------------------------

def extract_with_newspaper(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    article = Article(url)
    article.download()
    article.parse()

    text = article.text or ""
    title = article.title or ""
    pub = None
    try:
        if article.publish_date:
            pub = article.publish_date.isoformat()
    except Exception:
        pub = None
    return text, title, pub


def extract_with_trafilatura(
    url: str,
    session: requests.Session,
    timeout: Tuple[float, float],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if trafilatura is None:
        return None, None, None

    resp = session.get(url, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        return None, None, None

    downloaded = resp.text
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    title = None
    pub = extract_publish_date_from_html(downloaded)
    return text, title, pub


def extract_with_readability(
    url: str,
    session: requests.Session,
    timeout: Tuple[float, float],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    if Document is None or BeautifulSoup is None:
        return None, None, None

    resp = session.get(url, timeout=timeout)
    if not (200 <= resp.status_code < 300):
        return None, None, None

    html = resp.text
    doc = Document(html)
    title = (doc.short_title() or "").strip()

    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "html.parser")
    text = soup.get_text(separator="\n").strip()

    pub = extract_publish_date_from_html(html)
    return text, title, pub


def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def extract_article_content(
    row: dict,
    session: requests.Session,
    semaphores: Dict[str, "threading.Semaphore"],
    config: ScrapeConfig,
    logger: logging.Logger,
) -> dict:
    url = row.get("url") or ""
    domain = row.get("domain") or row.get("source") or ""
    seendate = row.get("seendate") or ""
    canonical_url = row.get("canonical_url") or canonicalize_url(url)

    if not url:
        return {"success": False, "url": url, "canonical_url": canonical_url, "domain": domain, "error": "missing_url"}

    sem = semaphores.get(domain)
    if sem is None:
        return {"success": False, "url": url, "canonical_url": canonical_url, "domain": domain, "error": "missing_semaphore"}

    timeout = (config.timeout_connect, config.timeout_read)
    time.sleep(0.05 + random.random() * 0.15)

    try:
        with sem:
            text = title = pub = None
            extractor_used = None

            try:
                t, ti, p = extract_with_newspaper(url)
                text, title, pub = t, ti, p
                extractor_used = "newspaper"
            except Exception:
                text = None

            if (not text) and trafilatura is not None:
                try:
                    t, ti, p = extract_with_trafilatura(url, session, timeout)
                    text = t
                    title = ti or title
                    pub = p or pub
                    extractor_used = "trafilatura"
                except Exception:
                    text = None

            if (not text) and Document is not None:
                try:
                    t, ti, p = extract_with_readability(url, session, timeout)
                    text = t
                    title = ti or title
                    pub = p or pub
                    extractor_used = "readability"
                except Exception:
                    text = None

            text = clean_text(text or "")
            title_final = (title or row.get("title") or "").strip()

            if len(text) < config.min_content_chars:
                return {
                    "success": False,
                    "url": url,
                    "canonical_url": canonical_url,
                    "domain": domain,
                    "status": "too_short_or_empty",
                    "extractor": extractor_used or "",
                }

            # publish_date fix (meta + JSON-LD) even when newspaper succeeds
            if not pub and BeautifulSoup is not None:
                try:
                    resp = session.get(url, timeout=timeout)
                    if 200 <= resp.status_code < 300 and resp.text:
                        pub2 = extract_publish_date_from_html(resp.text)
                        if pub2:
                            pub = pub2
                except Exception:
                    pass

            # NEW: human-readable datetime columns
            gdelt_dt = _parse_any_datetime(seendate)
            pub_dt = _parse_any_datetime(pub) if pub else None
            event_dt = pub_dt if pub_dt else gdelt_dt

            return {
                "success": True,
                "gdelt_seendate": seendate,
                "gdelt_datetime": _to_human(gdelt_dt),          # NEW human-readable
                "publish_date": pub or "",
                "publish_datetime": _to_human(pub_dt),          # NEW human-readable
                "event_datetime": _to_human(event_dt),          # NEW human-readable “best usable”
                "title": title_final,
                "content": text,
                "source": domain,
                "url": url,
                "canonical_url": canonical_url,
                "extractor": extractor_used or "",
                "content_hash": stable_hash(text),
            }

    except Exception as e:
        return {
            "success": False,
            "url": url,
            "canonical_url": canonical_url,
            "domain": domain,
            "error": f"{type(e).__name__}: {e}",
        }


# -----------------------------
# Parallel scrape
# -----------------------------

def scrape_parallel(
    gdelt_df: pd.DataFrame,
    session: requests.Session,
    config: ScrapeConfig,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import threading

    if gdelt_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    semaphores: Dict[str, threading.Semaphore] = {}
    for d in gdelt_df["domain"].fillna("").unique().tolist():
        semaphores[d] = threading.Semaphore(config.max_per_domain)

    rows = gdelt_df.to_dict("records")

    logger.info(f"Starting scraping: urls={len(rows)} workers={config.max_workers} per_domain={config.max_per_domain}")

    successes: List[dict] = []
    failures: List[dict] = []

    with ThreadPoolExecutor(max_workers=config.max_workers) as ex:
        future_to_row = {
            ex.submit(extract_article_content, row, session, semaphores, config, logger): row
            for row in rows
        }
        for fut in tqdm(as_completed(future_to_row), total=len(future_to_row)):
            res = fut.result()
            if res.get("success"):
                successes.append(res)
            else:
                failures.append(res)

    return pd.DataFrame(successes), pd.DataFrame(failures)


# -----------------------------
# Output helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_outputs(
    df: pd.DataFrame,
    failures: pd.DataFrame,
    config: ScrapeConfig,
    logger: logging.Logger,
    month_tag: str,
) -> Tuple[str, str]:
    ensure_dir(config.data_dir)

    base = f"oil_news_{month_tag}"
    out_main = os.path.join(config.data_dir, f"{base}.{config.output_format}")
    out_fail = os.path.join(config.data_dir, f"{base}_failures.csv")

    # UPDATED: include new human-readable datetime columns
    keep = [
        "gdelt_seendate", "gdelt_datetime",
        "publish_date", "publish_datetime",
        "event_datetime",
        "title", "content",
        "source", "url", "canonical_url",
        "extractor", "content_hash"
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = ""

    df = df[keep].copy()

    if "content_hash" in df.columns and not df.empty:
        before = len(df)
        df = df.drop_duplicates(subset=["content_hash"], keep="first")
        logger.info(f"Dedup by content_hash: {before} -> {len(df)}")

    if config.output_format.lower() == "parquet":
        try:
            df.to_parquet(out_main, index=False)
        except Exception as e:
            logger.warning(f"Parquet write failed ({e}); falling back to CSV.")
            out_main = os.path.join(config.data_dir, f"{base}.csv")
            df.to_csv(out_main, index=False)
    else:
        df.to_csv(out_main, index=False)

    if failures is None or failures.empty:
        failures = pd.DataFrame(columns=["url", "domain", "error", "status", "extractor"])
    failures.to_csv(out_fail, index=False)

    return out_main, out_fail


# -----------------------------
# Month processing
# -----------------------------

def month_date_range(year: int, month: int) -> Tuple[datetime, datetime]:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end


def process_month(config: ScrapeConfig) -> pd.DataFrame:
    start_date, end_date = month_date_range(config.year, config.month)

    gdelt_min = datetime(2017, 2, 19)
    if start_date < gdelt_min:
        start_date = gdelt_min

    month_tag = start_date.strftime("%Y_%m")

    ensure_dir(config.data_dir)
    logger = setup_logger(os.path.join(config.data_dir, f"scrape_{month_tag}.log"))
    session = build_session()

    logger.info("=" * 80)
    logger.info(f"PROCESSING {start_date.strftime('%B %Y')}")
    logger.info(f"Keywords: {config.keywords}")
    logger.info(f"Domains ({len(config.domains)}): {', '.join(config.domains)}")
    logger.info("=" * 80)

    gdelt_df = fetch_gdelt_urls_multi_domain(
        session=session,
        logger=logger,
        keywords=config.keywords,
        domains=config.domains,
        start_date=start_date,
        end_date=end_date,
        config=config,
    )

    if gdelt_df.empty:
        logger.warning("No URLs found from GDELT.")
        return pd.DataFrame()

    logger.info(f"Unique URLs after dedupe: {len(gdelt_df)}")
    logger.info("URLs per source:")
    try:
        logger.info("\n" + str(gdelt_df["domain"].value_counts()))
    except Exception:
        pass

    success_df, fail_df = scrape_parallel(gdelt_df, session, config, logger)
    logger.info(f"Extraction complete: success={len(success_df)} fail={len(fail_df)}")

    if success_df.empty:
        logger.warning("No articles extracted successfully.")
        _, out_fail = write_outputs(success_df, fail_df, config, logger, month_tag)
        logger.info(f"Saved failures: {out_fail}")
        return pd.DataFrame()

    out_main, out_fail = write_outputs(success_df, fail_df, config, logger, month_tag)
    logger.info(f"Saved articles: {out_main}")
    logger.info(f"Saved failures: {out_fail}")

    try:
        lens = success_df["content"].astype(str).str.len()
        logger.info(f"Content length mean={lens.mean():.0f} median={lens.median():.0f}")
        logger.info("Final per-source counts:")
        logger.info("\n" + str(success_df["source"].value_counts()))
    except Exception:
        pass

    return success_df


def iter_month_starts(start_year: int, start_month: int, end_year: int, end_month: int):
    y, m = start_year, start_month
    while (y < end_year) or (y == end_year and m <= end_month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    trusted_domains = [
        "oilprice.com",
        "reuters.com",
        "rigzone.com",
        "cnbc.com",
        "marketwatch.com",
    ]

    data_dir = "/Users/aswathsuresh/Documents/Projects/Campbell-B/data"
    keywords = "oil"

    START_YEAR, START_MONTH = 2022, 1
    END_YEAR, END_MONTH = 2025, 12

    max_workers = 10
    max_per_domain = 2

    all_month_summaries = []

    print("=" * 80)
    print(f"Extracting {keywords!r} news from {START_YEAR}-{START_MONTH:02d} to {END_YEAR}-{END_MONTH:02d}")
    print(f"Saving monthly files into: {data_dir}")
    print("=" * 80)

    for y, m in iter_month_starts(START_YEAR, START_MONTH, END_YEAR, END_MONTH):
        cfg = ScrapeConfig(
            data_dir=data_dir,
            keywords=keywords,
            domains=trusted_domains,
            year=y,
            month=m,
            max_workers=max_workers,
            max_per_domain=max_per_domain,
            output_format="csv",
        )

        df_month = process_month(cfg)

        all_month_summaries.append(
            {"year": y, "month": m, "articles_extracted": int(len(df_month))}
        )

        time.sleep(2.0 + random.random() * 2.0)

    summary_df = pd.DataFrame(all_month_summaries)
    summary_path = os.path.join(data_dir, "extraction_summary_2021_2025.csv")
    summary_df.to_csv(summary_path, index=False)

    print("=" * 80)
    print("DONE (2021–2025)")
    print(f"Monthly summary saved to: {summary_path}")
    print(summary_df)
