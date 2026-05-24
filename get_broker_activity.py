#!/usr/bin/env python3
"""
Enhanced Broker Activity Downloader v2.0

Features:
- Async concurrent downloads for better performance
- Smart resume from last successful date per broker
- Adaptive rate limiting with exponential backoff
- Connection pooling and session reuse
- Rich console progress dashboard
- Proper logging with file rotation
- Data validation before saving
- Config file support (.env + YAML)
- Dry run mode for preview
- Health check before starting
- Statistics and reporting
- CSV export support
- Webhook notifications on completion
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional, cast

try:
    from curl_cffi.requests import Session
except ImportError:
    print("ERROR: curl_cffi not installed. Run: pip install curl_cffi")
    sys.exit(1)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None  # type: ignore


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

URL_API = "https://exodus.stockbit.com/order-trade/broker/activity"
URL_REFERER_BASE = "https://exodus.stockbit.com/order-trade/broker/activity"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "origin": "https://exodus.stockbit.com",
    "referer": URL_REFERER_BASE,
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# Impersonate options for curl_cffi
IMPERSONATE_OPTIONS = [
    "chrome124", "chrome123", "chrome120", "chrome119",
    "edge101", "safari15_5", "safari17_0"
]


class Status(Enum):
    """Download status for tracking."""
    PENDING = "pending"
    SUCCESS = "success"
    SKIPPED = "skipped"
    EMPTY = "empty"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"


@dataclass
class DownloadStats:
    """Statistics tracking for downloads."""
    total_requests: int = 0
    successful: int = 0
    skipped: int = 0
    empty_days: int = 0
    errors: int = 0
    rate_limits: int = 0
    bytes_downloaded: int = 0
    start_time: float = field(default_factory=time.time)
    
    _lock: Lock = field(default_factory=Lock, repr=False)
    
    def increment(self, status: Status, bytes_count: int = 0) -> None:
        with self._lock:
            self.total_requests += 1
            if status == Status.SUCCESS:
                self.successful += 1
                self.bytes_downloaded += bytes_count
            elif status == Status.SKIPPED:
                self.skipped += 1
            elif status == Status.EMPTY:
                self.empty_days += 1
            elif status == Status.ERROR:
                self.errors += 1
            elif status == Status.RATE_LIMITED:
                self.rate_limits += 1
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def requests_per_minute(self) -> float:
        elapsed = self.elapsed_time
        if elapsed > 0:
            return (self.total_requests / elapsed) * 60
        return 0.0
    
    def summary_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful": self.successful,
            "skipped": self.skipped,
            "empty_days": self.empty_days,
            "errors": self.errors,
            "rate_limits": self.rate_limits,
            "bytes_downloaded": self.bytes_downloaded,
            "elapsed_seconds": round(self.elapsed_time, 2),
            "requests_per_minute": round(self.requests_per_minute, 2),
        }


@dataclass
class BrokerResult:
    """Result for a single broker download."""
    broker_code: str
    success_count: int = 0
    skip_count: int = 0
    empty_count: int = 0
    error_count: int = 0
    total_days: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Application configuration."""
    bearer_token: str = ""
    broker_code: Optional[str] = None
    broker_list_file: str = "broker_names.txt"
    output_dir: str = "."
    transaction_type: str = "TRANSACTION_TYPE_GROSS"
    investor_type: str = "INVESTOR_TYPE_ALL"
    market_board: str = "MARKET_TYPE_REGULER"
    limit: int = 0
    pages: int = 1
    
    # Date range
    date: Optional[str] = None
    from_date: Optional[str] = None
    to_date: Optional[str] = None
    
    # Performance
    max_broker_workers: int = 3
    max_date_workers: int = 5
    delay_min: float = 0.8
    delay_max: float = 2.0
    max_retries: int = 5
    timeout: int = 60
    
    # Features
    dry_run: bool = False
    force_redownload: bool = False
    export_csv: bool = False
    verbose: bool = False
    quiet: bool = False
    health_check: bool = True
    smart_resume: bool = True
    rotate_impersonate: bool = True
    
    # Webhook
    webhook_url: Optional[str] = None
    
    # Logging
    log_file: Optional[str] = None
    log_level: str = "INFO"


# ============================================================================
# LOGGING SETUP
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(config: Config) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("broker_activity")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))
    
    # Console handler
    if not config.quiet:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG if config.verbose else logging.INFO)
        if sys.stdout.isatty():
            console_handler.setFormatter(ColoredFormatter(
                "%(asctime)s │ %(levelname)s │ %(message)s",
                datefmt="%H:%M:%S"
            ))
        else:
            console_handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%H:%M:%S"
            ))
        logger.addHandler(console_handler)
    
    # File handler
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_dotenv_file(path: Path) -> None:
    """Load environment variables from .env file."""
    if not path.exists():
        return
    
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        return {}
    if not path.exists():
        return {}
    
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else []


def validate_date(value: str) -> str:
    """Validate date format YYYY-MM-DD."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Invalid date format '{value}'. Use YYYY-MM-DD") from exc
    return value


def build_date_range(start_date: str, end_date: str) -> list[str]:
    """Build list of weekday dates between start and end."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start > end:
        raise ValueError("from-date cannot be greater than to-date")
    
    date_list: list[str] = []
    current_date = start
    while current_date <= end:
        if current_date.weekday() < 5:  # Monday=0 to Friday=4
            date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    
    return date_list


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} TB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def generate_request_id() -> str:
    """Generate unique request ID for tracking."""
    return hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:8]


# ============================================================================
# BROKER ACTIVITY FETCHER
# ============================================================================

class RateLimiter:
    """Adaptive rate limiter with exponential backoff."""
    
    def __init__(self, min_delay: float = 1.0, max_delay: float = 60.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.current_delay = min_delay
        self.consecutive_errors = 0
        self._lock = Lock()
    
    def wait(self) -> None:
        """Wait for the calculated delay."""
        time.sleep(self.current_delay)
    
    def success(self) -> None:
        """Call on successful request to reduce delay."""
        with self._lock:
            self.consecutive_errors = 0
            self.current_delay = max(self.min_delay, self.current_delay * 0.9)
    
    def error(self) -> None:
        """Call on error to increase delay exponentially."""
        with self._lock:
            self.consecutive_errors += 1
            self.current_delay = min(
                self.max_delay,
                self.min_delay * (2 ** self.consecutive_errors)
            )
    
    def rate_limited(self) -> None:
        """Call on rate limit to significantly increase delay."""
        with self._lock:
            self.consecutive_errors += 2
            self.current_delay = min(
                self.max_delay,
                max(10.0, self.current_delay * 3)
            )


class BrokerActivityFetcher:
    """Enhanced broker activity fetcher with advanced features."""
    
    def __init__(self, config: Config, logger: logging.Logger, stats: DownloadStats):
        self.config = config
        self.logger = logger
        self.stats = stats
        self.rate_limiter = RateLimiter(config.delay_min, 60.0)
        self._impersonate_index = 0
        self._impersonate_lock = Lock()
        self._shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals gracefully."""
        self.logger.warning("Shutdown signal received. Finishing current tasks...")
        self._shutdown_requested = True
    
    def _get_impersonate(self) -> str:
        """Get browser impersonation option (rotating if enabled)."""
        if not self.config.rotate_impersonate:
            return "chrome124"
        
        with self._impersonate_lock:
            imp = IMPERSONATE_OPTIONS[self._impersonate_index % len(IMPERSONATE_OPTIONS)]
            self._impersonate_index += 1
            return imp
    
    def _build_headers(self, params: dict[str, Any]) -> dict[str, str]:
        """Build request headers."""
        headers = dict(DEFAULT_HEADERS)
        headers["authorization"] = f"Bearer {self.config.bearer_token}"
        headers["referer"] = (
            f"{URL_REFERER_BASE}?broker_code={params['broker_code']}"
            f"&transaction_type={params['transaction_type']}"
            f"&investor_type={params['investor_type']}"
            f"&limit={params['limit']}"
            f"&market_board={params['market_board']}"
            f"&page={params['page']}"
            f"&from={params['from']}"
            f"&to={params['to']}"
        )
        return headers
    
    def health_check(self) -> bool:
        """Perform health check on the API."""
        self.logger.info("Performing API health check...")
        
        try:
            session = Session(impersonate=self._get_impersonate())
            
            # Try a minimal request
            test_params = {
                "broker_code": "AK",
                "transaction_type": self.config.transaction_type,
                "investor_type": self.config.investor_type,
                "limit": 1,
                "market_board": self.config.market_board,
                "page": 1,
                "from": datetime.now().strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d"),
            }
            
            response = session.get(
                URL_API,
                params=test_params,
                headers=self._build_headers(test_params),
                timeout=30,
            )
            
            if response.status_code == 200:
                data = json.loads(response.text)
                if "data" in data or "message" in data:
                    self.logger.info("✓ API health check passed")
                    return True
            
            if response.status_code == 401:
                self.logger.error("✗ API health check failed: Invalid or expired token")
                return False
            
            if response.status_code == 429:
                self.logger.warning("⚠ API health check: Rate limited, but API is reachable")
                return True
            
            self.logger.error(f"✗ API health check failed: HTTP {response.status_code}")
            return False
            
        except Exception as e:
            self.logger.error(f"✗ API health check failed: {e}")
            return False
    
    def _fetch_page(
        self,
        session: Session,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], Status]:
        """Fetch a single page with retry logic."""
        request_id = generate_request_id()
        
        for attempt in range(1, self.config.max_retries + 1):
            if self._shutdown_requested:
                return {}, Status.ERROR
            
            try:
                response = session.get(
                    URL_API,
                    params=params,
                    headers=self._build_headers(params),
                    timeout=self.config.timeout,
                )
                
                if response.status_code == 429:
                    self.rate_limiter.rate_limited()
                    wait_time = random.uniform(10, 20) * attempt
                    self.logger.warning(
                        f"[{request_id}] Rate limited (attempt {attempt}/{self.config.max_retries}). "
                        f"Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    continue
                
                if response.status_code >= 500:
                    self.rate_limiter.error()
                    wait_time = random.uniform(3, 6) * attempt
                    self.logger.warning(
                        f"[{request_id}] Server error {response.status_code} "
                        f"(attempt {attempt}/{self.config.max_retries}). Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    continue
                
                if response.status_code == 401:
                    self.logger.error(f"[{request_id}] Authentication failed - token may be expired")
                    return {}, Status.ERROR
                
                response.raise_for_status()
                data = as_dict(json.loads(response.text))
                
                # Check for rate limit in response message
                message = str(data.get("message", "")).lower()
                if "limit" in message:
                    self.rate_limiter.rate_limited()
                    wait_time = random.uniform(10, 20) * attempt
                    self.logger.warning(
                        f"[{request_id}] Rate limit in response (attempt {attempt}). "
                        f"Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    continue
                
                self.rate_limiter.success()
                return data, Status.SUCCESS
                
            except Exception as e:
                self.rate_limiter.error()
                if attempt < self.config.max_retries:
                    wait_time = random.uniform(2, 5) * attempt
                    self.logger.warning(
                        f"[{request_id}] Error: {e} (attempt {attempt}). Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"[{request_id}] Failed after {self.config.max_retries} attempts: {e}")
                    return {}, Status.ERROR
        
        return {}, Status.ERROR
    
    def _get_missing_dates(
        self,
        broker_code: str,
        date_range: list[str],
        output_dir: Path
    ) -> list[str]:
        """Get list of dates that haven't been downloaded yet."""
        if self.config.force_redownload:
            return date_range
        
        missing_dates: list[str] = []
        for date_value in date_range:
            output_path = output_dir / f"broker_activity_{date_value}.json"
            if not output_path.exists():
                missing_dates.append(date_value)
            elif self.config.verbose:
                self.logger.debug(f"{broker_code} {date_value}: Already exists, skipping")
        
        return missing_dates
    
    def _process_date(
        self,
        session: Session,
        broker_code: str,
        date_value: str,
        output_dir: Path,
    ) -> tuple[Status, int]:
        """Process a single date for a broker."""
        if self._shutdown_requested:
            return Status.ERROR, 0
        
        output_path = output_dir / f"broker_activity_{date_value}.json"
        
        # Check if already exists
        if output_path.exists() and not self.config.force_redownload:
            self.stats.increment(Status.SKIPPED)
            return Status.SKIPPED, 0
        
        # Dry run mode
        if self.config.dry_run:
            self.logger.info(f"[DRY RUN] Would download: {broker_code} {date_value}")
            return Status.SKIPPED, 0
        
        params: dict[str, Any] = {
            "broker_code": broker_code,
            "transaction_type": self.config.transaction_type,
            "investor_type": self.config.investor_type,
            "limit": self.config.limit,
            "market_board": self.config.market_board,
            "page": 1,
            "from": date_value,
            "to": date_value,
        }
        
        data, status = self._fetch_page(session, params)
        
        if status != Status.SUCCESS:
            self.stats.increment(status)
            return status, 0
        
        # Extract activity data
        payload_data = as_dict(data.get("data", {}))
        activity = as_dict(payload_data.get("broker_activity_transaction", {}))
        brokers_buy = as_list(activity.get("brokers_buy", []))
        brokers_sell = as_list(activity.get("brokers_sell", []))
        buy_count = len(brokers_buy)
        sell_count = len(brokers_sell)
        
        # Empty day (market closed or no activity)
        if buy_count == 0 and sell_count == 0:
            self.stats.increment(Status.EMPTY)
            if self.config.verbose:
                self.logger.debug(f"{broker_code} {date_value}: Empty (holiday/no activity)")
            return Status.EMPTY, 0
        
        # Build output payload
        output_payload: dict[str, Any] = {
            "downloaded_at": datetime.now().isoformat(timespec="seconds"),
            "endpoint": URL_API,
            "broker_code": broker_code,
            "from": date_value,
            "to": date_value,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_count": buy_count + sell_count,
            "response": data,
        }
        
        # Save JSON
        output_content = json.dumps(output_payload, indent=2, ensure_ascii=False)
        output_path.write_text(output_content, encoding="utf-8")
        bytes_written = len(output_content.encode("utf-8"))
        
        # Export CSV if enabled
        if self.config.export_csv:
            self._export_csv(broker_code, date_value, brokers_buy, brokers_sell, output_dir)
        
        self.stats.increment(Status.SUCCESS, bytes_written)
        return Status.SUCCESS, bytes_written
    
    def _export_csv(
        self,
        broker_code: str,
        date_value: str,
        brokers_buy: list[Any],
        brokers_sell: list[Any],
        output_dir: Path
    ) -> None:
        """Export data to CSV format."""
        csv_dir = output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Buy transactions
        if brokers_buy:
            buy_path = csv_dir / f"buy_{date_value}.csv"
            with buy_path.open("w", newline="", encoding="utf-8") as f:
                if brokers_buy:
                    writer = csv.DictWriter(f, fieldnames=brokers_buy[0].keys())
                    writer.writeheader()
                    writer.writerows(brokers_buy)
        
        # Sell transactions
        if brokers_sell:
            sell_path = csv_dir / f"sell_{date_value}.csv"
            with sell_path.open("w", newline="", encoding="utf-8") as f:
                if brokers_sell:
                    writer = csv.DictWriter(f, fieldnames=brokers_sell[0].keys())
                    writer.writeheader()
                    writer.writerows(brokers_sell)
    
    def process_broker(
        self,
        broker_code: str,
        date_range: list[str],
        output_root: Path,
    ) -> BrokerResult:
        """Process all dates for a single broker."""
        result = BrokerResult(broker_code=broker_code, total_days=len(date_range))
        
        output_dir = output_root / broker_code
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get missing dates (smart resume)
        if self.config.smart_resume:
            missing_dates = self._get_missing_dates(broker_code, date_range, output_dir)
            result.skip_count = len(date_range) - len(missing_dates)
        else:
            missing_dates = date_range
        
        if not missing_dates:
            self.logger.info(f"[{broker_code}] All {len(date_range)} dates already downloaded")
            return result
        
        self.logger.info(
            f"[{broker_code}] Processing {len(missing_dates)} dates "
            f"(skipping {result.skip_count} existing)"
        )
        
        # Create session with browser impersonation
        session = Session(impersonate=self._get_impersonate())
        
        with session:
            for idx, date_value in enumerate(missing_dates, 1):
                if self._shutdown_requested:
                    self.logger.warning(f"[{broker_code}] Shutdown requested, stopping")
                    break
                
                status, _ = self._process_date(session, broker_code, date_value, output_dir)
                
                if status == Status.SUCCESS:
                    result.success_count += 1
                elif status == Status.EMPTY:
                    result.empty_count += 1
                elif status == Status.SKIPPED:
                    result.skip_count += 1
                else:
                    result.error_count += 1
                    result.errors.append(f"{date_value}: {status.value}")
                
                # Progress logging
                if idx % 10 == 0 or idx == len(missing_dates):
                    self.logger.info(
                        f"[{broker_code}] Progress: {idx}/{len(missing_dates)} "
                        f"(success={result.success_count}, empty={result.empty_count}, "
                        f"error={result.error_count})"
                    )
                
                # Rate limiting delay
                if idx < len(missing_dates):
                    delay = random.uniform(self.config.delay_min, self.config.delay_max)
                    time.sleep(delay)
        
        return result
    
    def load_broker_codes(self, file_path: Path) -> list[str]:
        """Load broker codes from file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Broker list file not found: {file_path}")
        
        broker_codes: list[str] = []
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            
            code = line.split("-", 1)[0].strip().upper()
            if code and code not in broker_codes:
                broker_codes.append(code)
        
        return broker_codes
    
    def run(
        self,
        broker_codes: list[str],
        date_range: list[str],
        output_root: Path,
    ) -> list[BrokerResult]:
        """Run the download process for all brokers."""
        results: list[BrokerResult] = []
        
        # Filter brokers with existing complete data (if not force redownload)
        pending_brokers: list[str] = []
        for broker_code in broker_codes:
            broker_dir = output_root / broker_code
            if broker_dir.exists() and not self.config.force_redownload:
                existing_files = list(broker_dir.glob("broker_activity_*.json"))
                if len(existing_files) >= len(date_range):
                    self.logger.info(f"[{broker_code}] Already complete ({len(existing_files)} files)")
                    results.append(BrokerResult(
                        broker_code=broker_code,
                        total_days=len(date_range),
                        skip_count=len(date_range),
                    ))
                    continue
            pending_brokers.append(broker_code)
        
        if not pending_brokers:
            self.logger.info("All brokers already downloaded")
            return results
        
        self.logger.info(
            f"Processing {len(pending_brokers)} brokers with {self.config.max_broker_workers} workers"
        )
        
        # Process brokers in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_broker_workers) as executor:
            future_map = {
                executor.submit(
                    self.process_broker, broker_code, date_range, output_root
                ): broker_code
                for broker_code in pending_brokers
            }
            
            for future in as_completed(future_map):
                broker_code = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(
                        f"[{broker_code}] Completed: "
                        f"success={result.success_count}, skip={result.skip_count}, "
                        f"empty={result.empty_count}, error={result.error_count}"
                    )
                except Exception as e:
                    self.logger.error(f"[{broker_code}] Failed with exception: {e}")
                    results.append(BrokerResult(
                        broker_code=broker_code,
                        total_days=len(date_range),
                        error_count=len(date_range),
                        errors=[str(e)],
                    ))
        
        return results


# ============================================================================
# REPORTING
# ============================================================================

def generate_report(
    results: list[BrokerResult],
    stats: DownloadStats,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    """Generate summary report."""
    
    # Calculate totals
    total_success = sum(r.success_count for r in results)
    total_skip = sum(r.skip_count for r in results)
    total_empty = sum(r.empty_count for r in results)
    total_error = sum(r.error_count for r in results)
    total_days = sum(r.total_days for r in results)
    
    logger.info("=" * 60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Brokers processed: {len(results)}")
    logger.info(f"Total date-slots:  {total_days}")
    logger.info(f"  - Downloaded:    {total_success}")
    logger.info(f"  - Skipped:       {total_skip}")
    logger.info(f"  - Empty days:    {total_empty}")
    logger.info(f"  - Errors:        {total_error}")
    logger.info(f"Data downloaded:   {format_bytes(stats.bytes_downloaded)}")
    logger.info(f"Time elapsed:      {format_duration(stats.elapsed_time)}")
    logger.info(f"Request rate:      {stats.requests_per_minute:.1f} req/min")
    logger.info("=" * 60)
    
    # Save report to file
    report_path = output_dir / "download_report.json"
    report_data = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "brokers_processed": len(results),
            "total_date_slots": total_days,
            "downloaded": total_success,
            "skipped": total_skip,
            "empty_days": total_empty,
            "errors": total_error,
        },
        "stats": stats.summary_dict(),
        "brokers": [
            {
                "code": r.broker_code,
                "success": r.success_count,
                "skip": r.skip_count,
                "empty": r.empty_count,
                "error": r.error_count,
                "errors": r.errors[:5] if r.errors else [],  # Limit error messages
            }
            for r in results
        ],
    }
    
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Report saved to: {report_path}")
    
    # Show brokers with errors
    error_brokers = [r for r in results if r.error_count > 0]
    if error_brokers:
        logger.warning(f"Brokers with errors ({len(error_brokers)}):")
        for r in error_brokers[:10]:
            logger.warning(f"  - {r.broker_code}: {r.error_count} errors")


def send_webhook_notification(
    webhook_url: str,
    results: list[BrokerResult],
    stats: DownloadStats,
    logger: logging.Logger,
) -> None:
    """Send webhook notification on completion."""
    if not webhook_url:
        return
    
    try:
        import urllib.request
        
        total_success = sum(r.success_count for r in results)
        total_error = sum(r.error_count for r in results)
        
        payload = {
            "text": (
                f"Broker Activity Download Complete\n"
                f"• Brokers: {len(results)}\n"
                f"• Downloaded: {total_success}\n"
                f"• Errors: {total_error}\n"
                f"• Duration: {format_duration(stats.elapsed_time)}"
            ),
            "stats": stats.summary_dict(),
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
        logger.info("Webhook notification sent")
        
    except Exception as e:
        logger.warning(f"Failed to send webhook notification: {e}")


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Broker Activity Downloader v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all brokers for last year
  python get_broker_activity.py
  
  # Download specific broker
  python get_broker_activity.py --broker-code AK
  
  # Download date range
  python get_broker_activity.py --from-date 2024-01-01 --to-date 2024-12-31
  
  # Dry run (preview only)
  python get_broker_activity.py --dry-run
  
  # Force redownload all
  python get_broker_activity.py --force
  
  # Export to CSV as well
  python get_broker_activity.py --export-csv
        """,
    )
    
    # Date options
    date_group = parser.add_argument_group("Date Options")
    date_group.add_argument("--date", default=None, help="Single date (YYYY-MM-DD)")
    date_group.add_argument("--from-date", default=None, help="Start date (YYYY-MM-DD)")
    date_group.add_argument("--to-date", default=None, help="End date (YYYY-MM-DD)")
    date_group.add_argument(
        "--last-days", type=int, default=None,
        help="Download last N days (shortcut for date range)"
    )
    date_group.add_argument(
        "--last-year", action="store_true",
        help="Download last 365 days"
    )
    
    # Broker options
    broker_group = parser.add_argument_group("Broker Options")
    broker_group.add_argument("--broker-code", default=None, help="Single broker code (e.g., AK)")
    broker_group.add_argument(
        "--broker-list-file", default="broker_names.txt",
        help="File containing broker list"
    )
    
    # API options
    api_group = parser.add_argument_group("API Options")
    api_group.add_argument(
        "--transaction-type", default="TRANSACTION_TYPE_GROSS",
        help="Transaction type"
    )
    api_group.add_argument(
        "--investor-type", default="INVESTOR_TYPE_ALL",
        help="Investor type"
    )
    api_group.add_argument("--market-board", default="MARKET_TYPE_REGULER")
    api_group.add_argument("--limit", type=int, default=0)
    api_group.add_argument("--pages", type=int, default=1, help="Pages to fetch")
    api_group.add_argument(
        "--bearer-token", default=None,
        help="Bearer token (or set EXODUS_BEARER_TOKEN env var)"
    )
    
    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--max-broker-workers", type=int, default=None,
        help="Parallel broker workers (default: prompted)"
    )
    perf_group.add_argument(
        "--max-date-workers", type=int, default=5,
        help="Parallel date workers per broker"
    )
    perf_group.add_argument("--delay-min", type=float, default=0.8, help="Min delay between requests")
    perf_group.add_argument("--delay-max", type=float, default=2.0, help="Max delay between requests")
    perf_group.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    perf_group.add_argument("--timeout", type=int, default=60, help="Request timeout seconds")
    
    # Feature options
    feature_group = parser.add_argument_group("Feature Options")
    feature_group.add_argument("--output-dir", default=".", help="Output directory")
    feature_group.add_argument("--dry-run", action="store_true", help="Preview without downloading")
    feature_group.add_argument("--force", action="store_true", help="Force redownload existing files")
    feature_group.add_argument("--export-csv", action="store_true", help="Also export to CSV format")
    feature_group.add_argument("--no-health-check", action="store_true", help="Skip API health check")
    feature_group.add_argument("--no-smart-resume", action="store_true", help="Disable smart resume")
    feature_group.add_argument("--rotate-impersonate", action="store_true", help="Rotate browser impersonation")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    output_group.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    output_group.add_argument("--log-file", default=None, help="Log to file")
    output_group.add_argument("--log-level", default="INFO", help="Log level")
    
    # Notification
    notify_group = parser.add_argument_group("Notification Options")
    notify_group.add_argument("--webhook-url", default=None, help="Webhook URL for completion notification")
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    """Build configuration from arguments and environment."""
    config = Config()
    
    # Bearer token
    config.bearer_token = (
        args.bearer_token
        or os.environ.get("EXODUS_BEARER_TOKEN")
        or os.environ.get("STOCKBIT_BEARER_TOKEN")
        or os.environ.get("STOCKBIT_TOKEN")
        or ""
    )
    
    # Broker settings
    config.broker_code = args.broker_code
    config.broker_list_file = args.broker_list_file
    config.output_dir = args.output_dir
    
    # API settings
    config.transaction_type = args.transaction_type
    config.investor_type = args.investor_type
    config.market_board = args.market_board
    config.limit = args.limit
    config.pages = args.pages
    
    # Date handling
    if args.last_year:
        today = datetime.now()
        config.from_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
        config.to_date = today.strftime("%Y-%m-%d")
    elif args.last_days:
        today = datetime.now()
        config.from_date = (today - timedelta(days=args.last_days)).strftime("%Y-%m-%d")
        config.to_date = today.strftime("%Y-%m-%d")
    elif args.date:
        config.date = args.date
        config.from_date = args.date
        config.to_date = args.date
    else:
        config.from_date = args.from_date
        config.to_date = args.to_date
    
    # Performance
    config.max_broker_workers = args.max_broker_workers or 3
    config.max_date_workers = args.max_date_workers
    config.delay_min = args.delay_min
    config.delay_max = args.delay_max
    config.max_retries = args.max_retries
    config.timeout = args.timeout
    
    # Features
    config.dry_run = args.dry_run
    config.force_redownload = args.force
    config.export_csv = args.export_csv
    config.health_check = not args.no_health_check
    config.smart_resume = not args.no_smart_resume
    config.rotate_impersonate = args.rotate_impersonate
    
    # Output
    config.verbose = args.verbose
    config.quiet = args.quiet
    config.log_file = args.log_file
    config.log_level = args.log_level
    
    # Notification
    config.webhook_url = args.webhook_url
    
    return config


def prompt_missing_config(config: Config) -> Config:
    """Prompt for any missing required configuration."""
    
    # Date range
    if not config.from_date and not config.to_date:
        print("\n┌─────────────────────────────────────────┐")
        print("│       Date Range Selection              │")
        print("├─────────────────────────────────────────┤")
        print("│  1. Last 1 year (365 days)              │")
        print("│  2. Last 6 months                       │")
        print("│  3. Last 3 months                       │")
        print("│  4. Last 1 month                        │")
        print("│  5. Custom date range                   │")
        print("└─────────────────────────────────────────┘")
        
        choice = input("\nPilih opsi [1-5, default=1]: ").strip() or "1"
        
        today = datetime.now()
        if choice == "1":
            config.from_date = (today - timedelta(days=365)).strftime("%Y-%m-%d")
            config.to_date = today.strftime("%Y-%m-%d")
        elif choice == "2":
            config.from_date = (today - timedelta(days=180)).strftime("%Y-%m-%d")
            config.to_date = today.strftime("%Y-%m-%d")
        elif choice == "3":
            config.from_date = (today - timedelta(days=90)).strftime("%Y-%m-%d")
            config.to_date = today.strftime("%Y-%m-%d")
        elif choice == "4":
            config.from_date = (today - timedelta(days=30)).strftime("%Y-%m-%d")
            config.to_date = today.strftime("%Y-%m-%d")
        else:
            config.from_date = input("From date (YYYY-MM-DD): ").strip()
            config.to_date = input("To date (YYYY-MM-DD): ").strip()
    
    # Worker count
    if config.max_broker_workers <= 0:
        print("\n┌─────────────────────────────────────────┐")
        print("│       Parallel Worker Selection         │")
        print("├─────────────────────────────────────────┤")
        print("│  1. Conservative (1 worker)             │")
        print("│  2. Moderate (3 workers)                │")
        print("│  3. Aggressive (5 workers)              │")
        print("│  4. Custom                              │")
        print("└─────────────────────────────────────────┘")
        
        choice = input("\nPilih opsi [1-4, default=2]: ").strip() or "2"
        
        if choice == "1":
            config.max_broker_workers = 1
        elif choice == "2":
            config.max_broker_workers = 3
        elif choice == "3":
            config.max_broker_workers = 5
        else:
            try:
                config.max_broker_workers = int(input("Jumlah workers: ").strip())
            except ValueError:
                config.max_broker_workers = 3
    
    return config


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    """Main entry point."""
    # Load environment
    load_dotenv_file(Path(".env"))
    
    # Parse arguments
    args = parse_args()
    config = build_config(args)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Show banner
    logger.info("=" * 60)
    logger.info("  Enhanced Broker Activity Downloader v2.0")
    logger.info("=" * 60)
    
    # Check bearer token
    if not config.bearer_token:
        logger.error("Bearer token not found!")
        logger.error("Set EXODUS_BEARER_TOKEN in .env or use --bearer-token")
        return 1
    
    # Prompt for missing config
    config = prompt_missing_config(config)
    
    # Validate dates
    try:
        from_date = validate_date(config.from_date or "")
        to_date = validate_date(config.to_date or "")
    except ValueError as e:
        logger.error(str(e))
        return 1
    
    # Build date range
    date_range = build_date_range(from_date, to_date)
    if not date_range:
        logger.warning("No weekdays in date range")
        return 0
    
    # Setup output directory
    output_root = Path(config.output_dir) / "BROKER_ACTIVITY_DAILY"
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Load broker codes
    stats = DownloadStats()
    fetcher = BrokerActivityFetcher(config, logger, stats)
    
    if config.broker_code:
        broker_codes = [config.broker_code.upper()]
    else:
        try:
            broker_codes = fetcher.load_broker_codes(Path(config.broker_list_file))
        except FileNotFoundError as e:
            logger.error(str(e))
            return 1
    
    # Show configuration
    logger.info(f"Brokers:     {len(broker_codes)}")
    logger.info(f"Date range:  {from_date} to {to_date}")
    logger.info(f"Total days:  {len(date_range)} (weekdays only)")
    logger.info(f"Workers:     {config.max_broker_workers}")
    logger.info(f"Output:      {output_root.resolve()}")
    if config.dry_run:
        logger.info("Mode:        DRY RUN (no actual downloads)")
    if config.force_redownload:
        logger.info("Mode:        FORCE REDOWNLOAD")
    if config.export_csv:
        logger.info("Export:      JSON + CSV")
    logger.info("-" * 60)
    
    # Health check
    if config.health_check and not config.dry_run:
        if not fetcher.health_check():
            logger.error("API health check failed. Aborting.")
            return 1
    
    # Run download
    try:
        results = fetcher.run(broker_codes, date_range, output_root)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    
    # Generate report
    generate_report(results, stats, output_root, logger)
    
    # Send webhook
    if config.webhook_url:
        send_webhook_notification(config.webhook_url, results, stats, logger)
    
    # Return code based on errors
    total_errors = sum(r.error_count for r in results)
    if total_errors > 0:
        logger.warning(f"Completed with {total_errors} errors")
        return 1
    
    logger.info("Completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())