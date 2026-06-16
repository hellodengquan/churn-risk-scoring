import json
import queue
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any, Generator

try:
    from sqlalchemy import (
        Column, String, Integer, Float, DateTime, create_engine, text
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

from .models import AccountBehavior


Base = declarative_base() if SQLALCHEMY_AVAILABLE else object


class AccountBehaviorDB:
    def __init__(
        self,
        db_url: str = "sqlite:///./churn_data.db",
    ):
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy is required for database access")

        self.engine = create_engine(db_url, echo=False)
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
