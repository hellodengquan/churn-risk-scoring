from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

try:
    from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, text
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .models import AccountBehavior


Base = declarative_base() if SQLALCHEMY_AVAILABLE else object


@dataclass
class QueryFallbackResult:
    total_records: int
    loaded_count: int
    used_fallback: bool
    fallback_reason: str
    pages_loaded: int
    execution_time_ms: float
    warning_messages: List[str]


class AccountBehaviorDB:
    def __init__(
        self,
        db_url: str = "sqlite:///./churn_data.db",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        query_timeout: int = 30,
        large_query_threshold: int = 10000,
        fallback_page_size: int = 1000,
        enable_fallback: bool = True,
    ):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for database access")

        self.db_url = db_url
        self.large_query_threshold = large_query_threshold
        self.fallback_page_size = fallback_page_size
        self.enable_fallback = enable_fallback
        self.query_timeout = query_timeout

        connect_args = {}
        if "sqlite" in db_url:
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = query_timeout
        elif "postgres" in db_url or "postgresql" in db_url:
            connect_args["connect_timeout"] = query_timeout
            connect_args["options"] = f"-c statement_timeout={query_timeout * 1000}"

        if "sqlite" in db_url:
            self.engine = create_engine(
                db_url,
                echo=False,
                connect_args=connect_args,
            )
        else:
            self.engine = create_engine(
                db_url,
                echo=False,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_recycle=pool_recycle,
                pool_pre_ping=pool_pre_ping,
                pool_use_lifo=True,
                connect_args=connect_args,
            )

        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self._init_tables()

    def _init_tables(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS account_behaviors (
                    account_id TEXT PRIMARY KEY,
                    login_count_last_30d INTEGER DEFAULT 0,
                    login_count_last_7d INTEGER DEFAULT 0,
                    last_login_days_ago INTEGER DEFAULT 999,
                    total_transactions_last_30d INTEGER DEFAULT 0,
                    total_transactions_last_90d INTEGER DEFAULT 0,
                    transaction_amount_last_30d REAL DEFAULT 0.0,
                    avg_session_minutes REAL DEFAULT 0.0,
                    support_tickets_last_30d INTEGER DEFAULT 0,
                    refund_count_last_90d INTEGER DEFAULT 0,
                    payment_failures_last_30d INTEGER DEFAULT 0,
                    feature_usage_count TEXT DEFAULT '{}',
                    email_open_rate REAL DEFAULT 0.0,
                    subscription_age_days INTEGER DEFAULT 0,
                    plan_level TEXT DEFAULT 'basic',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_last_login
                ON account_behaviors(last_login_days_ago)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_plan_level
                ON account_behaviors(plan_level)
            """))
            conn.commit()

    def upsert_account(self, account: AccountBehavior) -> None:
        feature_json = json.dumps(account.feature_usage_count, ensure_ascii=False)
        with self.engine.connect() as conn:
            conn.execute(text("""
                INSERT OR REPLACE INTO account_behaviors (
                    account_id, login_count_last_30d, login_count_last_7d,
                    last_login_days_ago, total_transactions_last_30d,
                    total_transactions_last_90d, transaction_amount_last_30d,
                    avg_session_minutes, support_tickets_last_30d,
                    refund_count_last_90d, payment_failures_last_30d,
                    feature_usage_count, email_open_rate,
                    subscription_age_days, plan_level, updated_at
                ) VALUES (
                    :account_id, :login_30d, :login_7d, :last_login,
                    :tx_30d, :tx_90d, :tx_amount, :session,
                    :tickets, :refunds, :pay_fail, :feature_json,
                    :email_rate, :sub_age, :plan, CURRENT_TIMESTAMP
                )
            """), {
                "account_id": account.account_id,
                "login_30d": account.login_count_last_30d,
                "login_7d": account.login_count_last_7d,
                "last_login": account.last_login_days_ago,
                "tx_30d": account.total_transactions_last_30d,
                "tx_90d": account.total_transactions_last_90d,
                "tx_amount": account.transaction_amount_last_30d,
                "session": account.avg_session_minutes,
                "tickets": account.support_tickets_last_30d,
                "refunds": account.refund_count_last_90d,
                "pay_fail": account.payment_failures_last_30d,
                "feature_json": feature_json,
                "email_rate": account.email_open_rate,
                "sub_age": account.subscription_age_days,
                "plan": account.plan_level,
            })
            conn.commit()

    def upsert_batch(self, accounts: List[AccountBehavior]) -> int:
        for account in accounts:
            self.upsert_account(account)
        return len(accounts)

    def load_all_accounts(
        self,
        plan_filter: Optional[str] = None,
        min_last_login: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[AccountBehavior]:
        query = "SELECT * FROM account_behaviors WHERE 1=1"
        params = {}

        if plan_filter:
            query += " AND plan_level = :plan"
            params["plan"] = plan_filter
        if min_last_login is not None:
            query += " AND last_login_days_ago >= :min_login"
            params["min_login"] = min_last_login

        query += " ORDER BY last_login_days_ago DESC"

        if limit:
            query += " LIMIT :limit"
            params["limit"] = limit

        accounts = []
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params)
            for row in result.mappings():
                try:
                    features = json.loads(row["feature_usage_count"]) if row["feature_usage_count"] else {}
                except (json.JSONDecodeError, TypeError):
                    features = {}

                account = AccountBehavior(
                    account_id=row["account_id"],
                    login_count_last_30d=row["login_count_last_30d"] or 0,
                    login_count_last_7d=row["login_count_last_7d"] or 0,
                    last_login_days_ago=row["last_login_days_ago"] or 999,
                    total_transactions_last_30d=row["total_transactions_last_30d"] or 0,
                    total_transactions_last_90d=row["total_transactions_last_90d"] or 0,
                    transaction_amount_last_30d=row["transaction_amount_last_30d"] or 0.0,
                    avg_session_minutes=row["avg_session_minutes"] or 0.0,
                    support_tickets_last_30d=row["support_tickets_last_30d"] or 0,
                    refund_count_last_90d=row["refund_count_last_90d"] or 0,
                    payment_failures_last_30d=row["payment_failures_last_30d"] or 0,
                    feature_usage_count=features,
                    email_open_rate=row["email_open_rate"] or 0.0,
                    subscription_age_days=row["subscription_age_days"] or 0,
                    plan_level=row["plan_level"] or "basic",
                )
                accounts.append(account)

        return accounts

    def get_account_count(self) -> int:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM account_behaviors"))
            return int(result.scalar() or 0)

    def delete_account(self, account_id: str) -> bool:
        with self.engine.connect() as conn:
            result = conn.execute(
                text("DELETE FROM account_behaviors WHERE account_id = :id"),
                {"id": account_id}
            )
            conn.commit()
            return result.rowcount > 0

    def upsert_batch(self, accounts: List[AccountBehavior]) -> int:
        if not accounts:
            return 0
        if len(accounts) == 1:
            self.upsert_account(accounts[0])
            return 1

        feature_jsons = [json.dumps(a.feature_usage_count, ensure_ascii=False) for a in accounts]
        rows = [
            (
                a.account_id,
                a.login_count_last_30d,
                a.login_count_last_7d,
                a.last_login_days_ago,
                a.total_transactions_last_30d,
                a.total_transactions_last_90d,
                a.transaction_amount_last_30d,
                a.avg_session_minutes,
                a.support_tickets_last_30d,
                a.refund_count_last_90d,
                a.payment_failures_last_30d,
                feature_jsons[i],
                a.email_open_rate,
                a.subscription_age_days,
                a.plan_level,
            )
            for i, a in enumerate(accounts)
        ]

        with self.engine.connect() as conn:
            if "sqlite" in self.db_url:
                conn.execute(text("""
                    INSERT OR REPLACE INTO account_behaviors (
                        account_id, login_count_last_30d, login_count_last_7d,
                        last_login_days_ago, total_transactions_last_30d,
                        total_transactions_last_90d, transaction_amount_last_30d,
                        avg_session_minutes, support_tickets_last_30d,
                        refund_count_last_90d, payment_failures_last_30d,
                        feature_usage_count, email_open_rate,
                        subscription_age_days, plan_level
                    ) VALUES (
                        :1, :2, :3, :4, :5, :6, :7, :8, :9, :10, :11, :12, :13, :14, :15
                    )
                """), rows)
            elif "postgres" in self.db_url or "postgresql" in self.db_url:
                try:
                    from sqlalchemy.dialects.postgresql import insert
                    metadata = Base.metadata
                    table = metadata.tables["account_behaviors"]
                    stmt = insert(table).values([
                        {
                            "account_id": a.account_id,
                            "login_count_last_30d": a.login_count_last_30d,
                            "login_count_last_7d": a.login_count_last_7d,
                            "last_login_days_ago": a.last_login_days_ago,
                            "total_transactions_last_30d": a.total_transactions_last_30d,
                            "total_transactions_last_90d": a.total_transactions_last_90d,
                            "transaction_amount_last_30d": a.transaction_amount_last_30d,
                            "avg_session_minutes": a.avg_session_minutes,
                            "support_tickets_last_30d": a.support_tickets_last_30d,
                            "refund_count_last_90d": a.refund_count_last_90d,
                            "payment_failures_last_30d": a.payment_failures_last_30d,
                            "feature_usage_count": feature_jsons[i],
                            "email_open_rate": a.email_open_rate,
                            "subscription_age_days": a.subscription_age_days,
                            "plan_level": a.plan_level,
                        }
                        for i, a in enumerate(accounts)
                    ])
                    stmt = stmt.on_conflict_do_update(
                        index_elements=["account_id"],
                        set_={c.name: c for c in stmt.excluded if c.name != "account_id"},
                    )
                    conn.execute(stmt)
                except Exception:
                    for a in accounts:
                        self.upsert_account(a)
            else:
                for a in accounts:
                    self.upsert_account(a)

            conn.commit()
        return len(accounts)

    def estimate_count(
        self,
        plan_filter: Optional[str] = None,
        min_last_login: Optional[int] = None,
    ) -> int:
        try:
            query = "SELECT COUNT(*) FROM account_behaviors WHERE 1=1"
            params: Dict[str, Any] = {}
            if plan_filter:
                query += " AND plan_level = :plan"
                params["plan"] = plan_filter
            if min_last_login is not None:
                query += " AND last_login_days_ago >= :min_login"
                params["min_login"] = min_last_login

            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                return int(result.scalar() or 0)
        except Exception:
            return -1

    def load_all_accounts_large(
        self,
        plan_filter: Optional[str] = None,
        min_last_login: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[List[AccountBehavior], QueryFallbackResult]:
        start_time = time.time()
        warnings: List[str] = []
        used_fallback = False
        fallback_reason = ""
        pages_loaded = 0

        estimated_count = self.estimate_count(plan_filter, min_last_login)
        if estimated_count >= 0 and estimated_count > self.large_query_threshold:
            warnings.append(
                f"预估记录数 {estimated_count} 超过阈值 {self.large_query_threshold}，启用分页退路"
            )
            used_fallback = True
            fallback_reason = f"large_result_set_{estimated_count}"

        if not self.enable_fallback or (estimated_count >= 0 and estimated_count <= self.large_query_threshold):
            try:
                accounts = self.load_all_accounts(
                    plan_filter=plan_filter,
                    min_last_login=min_last_login,
                    limit=limit,
                )
                if limit and len(accounts) > limit:
                    accounts = accounts[:limit]

                exec_time = (time.time() - start_time) * 1000
                return accounts, QueryFallbackResult(
                    total_records=estimated_count if estimated_count >= 0 else len(accounts),
                    loaded_count=len(accounts),
                    used_fallback=used_fallback,
                    fallback_reason=fallback_reason,
                    pages_loaded=1,
                    execution_time_ms=round(exec_time, 2),
                    warning_messages=warnings,
                )
            except Exception as e:
                if not self.enable_fallback:
                    raise
                warnings.append(f"主查询失败，启用分页退路: {str(e)}")
                used_fallback = True
                fallback_reason = f"query_failure_{type(e).__name__}"

        all_accounts: List[AccountBehavior] = []
        page_size = self.fallback_page_size
        offset = 0
        max_pages = 100

        try:
            while pages_loaded < max_pages:
                page_limit = min(page_size, (limit - len(all_accounts)) if limit else page_size)
                if limit and len(all_accounts) >= limit:
                    break

                query = """
                    SELECT * FROM account_behaviors WHERE 1=1
                """
                params: Dict[str, Any] = {}
                if plan_filter:
                    query += " AND plan_level = :plan"
                    params["plan"] = plan_filter
                if min_last_login is not None:
                    query += " AND last_login_days_ago >= :min_login"
                    params["min_login"] = min_last_login
                query += " ORDER BY last_login_days_ago DESC"
                query += " LIMIT :limit OFFSET :offset"
                params["limit"] = page_limit
                params["offset"] = offset

                with self.engine.connect() as conn:
                    result = conn.execute(text(query), params)
                    rows = result.mappings().fetchall()

                if not rows:
                    break

                for row in rows:
                    try:
                        features = json.loads(row["feature_usage_count"]) if row.get("feature_usage_count") else {}
                    except (json.JSONDecodeError, TypeError):
                        features = {}

                    all_accounts.append(AccountBehavior(
                        account_id=row["account_id"],
                        login_count_last_30d=int(row.get("login_count_last_30d", 0)),
                        login_count_last_7d=int(row.get("login_count_last_7d", 0)),
                        last_login_days_ago=int(row.get("last_login_days_ago", 999)),
                        total_transactions_last_30d=int(row.get("total_transactions_last_30d", 0)),
                        total_transactions_last_90d=int(row.get("total_transactions_last_90d", 0)),
                        transaction_amount_last_30d=float(row.get("transaction_amount_last_30d", 0.0)),
                        avg_session_minutes=float(row.get("avg_session_minutes", 0.0)),
                        support_tickets_last_30d=int(row.get("support_tickets_last_30d", 0)),
                        refund_count_last_90d=int(row.get("refund_count_last_90d", 0)),
                        payment_failures_last_30d=int(row.get("payment_failures_last_30d", 0)),
                        feature_usage_count=features,
                        email_open_rate=float(row.get("email_open_rate", 0.0)),
                        subscription_age_days=int(row.get("subscription_age_days", 0)),
                        plan_level=row.get("plan_level", "basic"),
                    ))

                pages_loaded += 1
                offset += page_size

                if len(rows) < page_limit:
                    break

        except Exception as e:
            warnings.append(f"分页查询在第 {pages_loaded} 页中断: {str(e)}")
            if not all_accounts:
                raise

        exec_time = (time.time() - start_time) * 1000

        if pages_loaded >= max_pages:
            warnings.append(f"达到最大页数限制 {max_pages}，可能存在未加载数据")

        return all_accounts, QueryFallbackResult(
            total_records=estimated_count if estimated_count >= 0 else len(all_accounts),
            loaded_count=len(all_accounts),
            used_fallback=True,
            fallback_reason=fallback_reason or "pagination_fallback",
            pages_loaded=pages_loaded,
            execution_time_ms=round(exec_time, 2),
            warning_messages=warnings,
        )

    def stream_accounts(
        self,
        plan_filter: Optional[str] = None,
        min_last_login: Optional[int] = None,
        chunk_size: int = 1000,
    ) -> Generator[AccountBehavior, None, None]:
        offset = 0
        while True:
            query = """
                SELECT * FROM account_behaviors WHERE 1=1
            """
            params: Dict[str, Any] = {}
            if plan_filter:
                query += " AND plan_level = :plan"
                params["plan"] = plan_filter
            if min_last_login is not None:
                query += " AND last_login_days_ago >= :min_login"
                params["min_login"] = min_last_login
            query += " ORDER BY last_login_days_ago DESC"
            query += " LIMIT :limit OFFSET :offset"
            params["limit"] = chunk_size
            params["offset"] = offset

            with self.engine.connect() as conn:
                result = conn.execute(text(query), params)
                rows = result.mappings().fetchall()

            if not rows:
                break

            for row in rows:
                try:
                    features = json.loads(row["feature_usage_count"]) if row.get("feature_usage_count") else {}
                except (json.JSONDecodeError, TypeError):
                    features = {}

                yield AccountBehavior(
                    account_id=row["account_id"],
                    login_count_last_30d=int(row.get("login_count_last_30d", 0)),
                    login_count_last_7d=int(row.get("login_count_last_7d", 0)),
                    last_login_days_ago=int(row.get("last_login_days_ago", 999)),
                    total_transactions_last_30d=int(row.get("total_transactions_last_30d", 0)),
                    total_transactions_last_90d=int(row.get("total_transactions_last_90d", 0)),
                    transaction_amount_last_30d=float(row.get("transaction_amount_last_30d", 0.0)),
                    avg_session_minutes=float(row.get("avg_session_minutes", 0.0)),
                    support_tickets_last_30d=int(row.get("support_tickets_last_30d", 0)),
                    refund_count_last_90d=int(row.get("refund_count_last_90d", 0)),
                    payment_failures_last_30d=int(row.get("payment_failures_last_30d", 0)),
                    feature_usage_count=features,
                    email_open_rate=float(row.get("email_open_rate", 0.0)),
                    subscription_age_days=int(row.get("subscription_age_days", 0)),
                    plan_level=row.get("plan_level", "basic"),
                )

            offset += chunk_size
            if len(rows) < chunk_size:
                break

    def check_pool_health(self) -> Dict[str, Any]:
        pool = getattr(self.engine.pool, 'pool', None)
        status = {
            "db_url": self.db_url,
            "pool_status": "unknown",
            "checked_in": 0,
            "checked_out": 0,
            "overflow": 0,
        }
        try:
            if pool is not None:
                status["checked_in"] = pool.checkedin() if hasattr(pool, 'checkedin') else 0
                status["checked_out"] = pool.checkedout() if hasattr(pool, 'checkedout') else 0
                status["overflow"] = pool.overflow() if hasattr(pool, 'overflow') else 0
                status["pool_status"] = "healthy"
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            status["connectivity"] = "ok"
        except Exception as e:
            status["connectivity"] = f"error: {str(e)}"
            status["pool_status"] = "unhealthy"
        return status


class StreamProcessor:
    def __init__(
        self,
        db: Optional[AccountBehaviorDB] = None,
        callback: Optional[Callable[[AccountBehavior], None]] = None,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.db = db
        self.callback = callback
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._queue: "queue.Queue[AccountBehavior]" = queue.Queue()
        self._batch: List[AccountBehavior] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._total_processed = 0
        self._total_errors = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2 * self.flush_interval)
        self.flush()

    def ingest(self, account: AccountBehavior) -> None:
        self._queue.put(account)

    def ingest_from_json(self, json_str: str) -> None:
        try:
            data = json.loads(json_str)
            account = AccountBehavior(**data)
            self.ingest(account)
        except Exception:
            with self._lock:
                self._total_errors += 1

    def ingest_batch(self, accounts: List[AccountBehavior]) -> None:
        for acc in accounts:
            self._queue.put(acc)

    def flush(self) -> int:
        with self._lock:
            count = len(self._batch)
            if count > 0:
                if self.db:
                    try:
                        self.db.upsert_batch(self._batch)
                    except Exception:
                        self._total_errors += count
                if self.callback:
                    for acc in self._batch:
                        try:
                            self.callback(acc)
                        except Exception:
                            self._total_errors += 1
                self._total_processed += count
                self._batch.clear()
            return count

    def _run_loop(self) -> None:
        last_flush = time.time()
        while self._running or not self._queue.empty():
            try:
                account = self._queue.get(timeout=0.5)
                with self._lock:
                    self._batch.append(account)

                with self._lock:
                    should_flush = len(self._batch) >= self.batch_size
                if should_flush:
                    self.flush()
                    last_flush = time.time()
            except queue.Empty:
                pass

            if time.time() - last_flush > self.flush_interval:
                self.flush()
                last_flush = time.time()

        self.flush()

    @property
    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "processed": self._total_processed,
                "errors": self._total_errors,
                "pending_queue": self._queue.qsize(),
                "pending_batch": len(self._batch),
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class KafkaStreamAdapter:
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "account-events",
        group_id: str = "churn-scoring-group",
    ):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._available = False
        try:
            import kafka
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def consume(
        self,
        processor: StreamProcessor,
        timeout_ms: int = 1000,
    ) -> Generator[AccountBehavior, None, None]:
        if not self._available:
            raise ImportError("kafka-python is required for Kafka integration")

        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        try:
            for message in consumer:
                try:
                    account = AccountBehavior(**message.value)
                    processor.ingest(account)
                    yield account
                except Exception:
                    continue
        finally:
            consumer.close()


def load_from_database(
    db_url: str,
    plan_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[AccountBehavior]:
    db = AccountBehaviorDB(db_url=db_url)
    return db.load_all_accounts(plan_filter=plan_filter, limit=limit)


@dataclass
class QueryFallbackResult:
    total_records: int
    loaded_count: int
    used_fallback: bool
    fallback_reason: str
    pages_loaded: int
    execution_time_ms: float
    warning_messages: List[str]


def create_data_source(
    source_type: str = "sqlite",
    **kwargs,
):
    source_type = source_type.lower()
    if source_type in ("sqlite", "sqlalchemy", "postgres", "mysql", "postgresql"):
        return AccountBehaviorDB(db_url=kwargs.get("db_url", "sqlite:///./churn_data.db"))
    elif source_type in ("mongodb", "mongo"):
        return MongoDBSource(**kwargs)
    elif source_type in ("clickhouse", "ch"):
        return ClickHouseSource(**kwargs)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}. Use 'sqlite', 'mongodb', or 'clickhouse'.")




class MongoDBSource:
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database: str = "churn_risk",
        collection: str = "account_behaviors",
    ):
        self.connection_string = connection_string
        self.database_name = database
        self.collection_name = collection
        self._client = None
        self._db = None
        self._coll = None
        self._available = False
        try:
            import pymongo
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def _connect(self):
        if not self._available:
            raise ImportError("pymongo is required for MongoDB integration")
        import pymongo
        if self._client is None:
            self._client = pymongo.MongoClient(self.connection_string)
            self._db = self._client[self.database_name]
            self._coll = self._db[self.collection_name]

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    def upsert_account(self, account: AccountBehavior) -> None:
        self._connect()
        data = asdict(account)
        self._coll.update_one(
            {"account_id": account.account_id},
            {"$set": data},
            upsert=True,
        )

    def upsert_batch(self, accounts: List[AccountBehavior]) -> int:
        if not accounts:
            return 0
        self._connect()
        from pymongo import UpdateOne
        operations = [
            UpdateOne(
                {"account_id": a.account_id},
                {"$set": asdict(a)},
                upsert=True,
            )
            for a in accounts
        ]
        result = self._coll.bulk_write(operations)
        return result.upserted_count + result.modified_count

    def load_all_accounts(
        self,
        plan_filter: Optional[str] = None,
        min_last_login: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[AccountBehavior]:
        self._connect()
        query: Dict[str, Any] = {}
        if plan_filter:
            query["plan_level"] = plan_filter
        if min_last_login is not None:
            query["last_login_days_ago"] = {"$gte": min_last_login}

        cursor = self._coll.find(query).sort("last_login_days_ago", -1)
        if limit:
            cursor = cursor.limit(limit)

        accounts = []
        for doc in cursor:
            doc.pop("_id", None)
            accounts.append(AccountBehavior(**doc))
        return accounts

    def get_account_count(self) -> int:
        self._connect()
        return int(self._coll.count_documents({}))

    def delete_account(self, account_id: str) -> bool:
        self._connect()
        result = self._coll.delete_one({"account_id": account_id})
        return result.deleted_count > 0


class ClickHouseSource:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8123,
        database: str = "churn_risk",
        username: str = "default",
        password: str = "",
        table: str = "account_behaviors",
    ):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.table = table
        self._client = None
        self._available = False
        try:
            import clickhouse_connect
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def _connect(self):
        if not self._available:
            raise ImportError("clickhouse-connect is required for ClickHouse integration")
        import clickhouse_connect
        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
            )
            self._init_table()

    def _init_table(self):
        self._client.command(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                account_id String,
                login_count_last_30d Int32,
                login_count_last_7d Int32,
                last_login_days_ago Int32,
                total_transactions_last_30d Int32,
                total_transactions_last_90d Int32,
                transaction_amount_last_30d Float64,
                avg_session_minutes Float64,
                support_tickets_last_30d Int32,
                refund_count_last_90d Int32,
                payment_failures_last_30d Int32,
                feature_usage_count String,
                email_open_rate Float64,
                subscription_age_days Int32,
                plan_level String,
                updated_at DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            ORDER BY (account_id)
        """)

    def close(self):
        if self._client:
            self._client.close()
            self._client = None

    def upsert_account(self, account: AccountBehavior) -> None:
        self._connect()
        feature_json = json.dumps(account.feature_usage_count, ensure_ascii=False)
        self._client.insert(
            self.table,
            [[
                account.account_id,
                account.login_count_last_30d,
                account.login_count_last_7d,
                account.last_login_days_ago,
                account.total_transactions_last_30d,
                account.total_transactions_last_90d,
                account.transaction_amount_last_30d,
                account.avg_session_minutes,
                account.support_tickets_last_30d,
                account.refund_count_last_90d,
                account.payment_failures_last_30d,
                feature_json,
                account.email_open_rate,
                account.subscription_age_days,
                account.plan_level,
            ]],
            column_names=[
                "account_id", "login_count_last_30d", "login_count_last_7d",
                "last_login_days_ago", "total_transactions_last_30d",
                "total_transactions_last_90d", "transaction_amount_last_30d",
                "avg_session_minutes", "support_tickets_last_30d",
                "refund_count_last_90d", "payment_failures_last_30d",
                "feature_usage_count", "email_open_rate",
                "subscription_age_days", "plan_level",
            ],
        )

    def upsert_batch(self, accounts: List[AccountBehavior]) -> int:
        if not accounts:
            return 0
        self._connect()
        rows = []
        for account in accounts:
            rows.append([
                account.account_id,
                account.login_count_last_30d,
                account.login_count_last_7d,
                account.last_login_days_ago,
                account.total_transactions_last_30d,
                account.total_transactions_last_90d,
                account.transaction_amount_last_30d,
                account.avg_session_minutes,
                account.support_tickets_last_30d,
                account.refund_count_last_90d,
                account.payment_failures_last_30d,
                json.dumps(account.feature_usage_count, ensure_ascii=False),
                account.email_open_rate,
                account.subscription_age_days,
                account.plan_level,
            ])
        self._client.insert(self.table, rows, column_names=[
            "account_id", "login_count_last_30d", "login_count_last_7d",
            "last_login_days_ago", "total_transactions_last_30d",
            "total_transactions_last_90d", "transaction_amount_last_30d",
            "avg_session_minutes", "support_tickets_last_30d",
            "refund_count_last_90d", "payment_failures_last_30d",
            "feature_usage_count", "email_open_rate",
            "subscription_age_days", "plan_level",
        ])
        return len(accounts)

    def load_all_accounts(
        self,
        plan_filter: Optional[str] = None,
        min_last_login: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[AccountBehavior]:
        self._connect()
        where_clauses = []
        params: Dict[str, Any] = {}
        if plan_filter:
            where_clauses.append("plan_level = {plan:String}")
            params["plan"] = plan_filter
        if min_last_login is not None:
            where_clauses.append("last_login_days_ago >= {min_login:Int32}")
            params["min_login"] = min_last_login

        query = f"SELECT * FROM {self.table}"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY last_login_days_ago DESC"
        if limit:
            query += " LIMIT {limit:Int32}"
            params["limit"] = limit

        result = self._client.query(query, parameters=params)
        accounts = []
        for row in result.named_results():
            try:
                features = json.loads(row["feature_usage_count"]) if row.get("feature_usage_count") else {}
            except (json.JSONDecodeError, TypeError):
                features = {}

            accounts.append(AccountBehavior(
                account_id=row["account_id"],
                login_count_last_30d=int(row.get("login_count_last_30d", 0)),
                login_count_last_7d=int(row.get("login_count_last_7d", 0)),
                last_login_days_ago=int(row.get("last_login_days_ago", 999)),
                total_transactions_last_30d=int(row.get("total_transactions_last_30d", 0)),
                total_transactions_last_90d=int(row.get("total_transactions_last_90d", 0)),
                transaction_amount_last_30d=float(row.get("transaction_amount_last_30d", 0.0)),
                avg_session_minutes=float(row.get("avg_session_minutes", 0.0)),
                support_tickets_last_30d=int(row.get("support_tickets_last_30d", 0)),
                refund_count_last_90d=int(row.get("refund_count_last_90d", 0)),
                payment_failures_last_30d=int(row.get("payment_failures_last_30d", 0)),
                feature_usage_count=features,
                email_open_rate=float(row.get("email_open_rate", 0.0)),
                subscription_age_days=int(row.get("subscription_age_days", 0)),
                plan_level=row.get("plan_level", "basic"),
            ))
        return accounts

    def get_account_count(self) -> int:
        self._connect()
        result = self._client.query(f"SELECT count() FROM {self.table}")
        return int(result.result_rows[0][0]) if result.result_rows else 0

    def delete_account(self, account_id: str) -> bool:
        self._connect()
        self._client.command(
            f"ALTER TABLE {self.table} DELETE WHERE account_id = {{id:String}}",
            parameters={"id": account_id},
        )
        return True


def create_data_source(
    source_type: str = "sqlite",
    **kwargs,
):
    source_type = source_type.lower()
    if source_type in ("sqlite", "sqlalchemy", "postgres", "mysql", "postgresql"):
        return AccountBehaviorDB(db_url=kwargs.get("db_url", "sqlite:///./churn_data.db"))
    elif source_type in ("mongodb", "mongo"):
        return MongoDBSource(**kwargs)
    elif source_type in ("clickhouse", "ch"):
        return ClickHouseSource(**kwargs)
    else:
        raise ValueError(f"Unsupported data source type: {source_type}. Use 'sqlite', 'mongodb', or 'clickhouse'.")

