import csv
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .models import AccountBehavior


def load_accounts_from_csv(file_path: str) -> List[AccountBehavior]:
    df = pd.read_csv(file_path)
    accounts = []
    for _, row in df.iterrows():
        feature_usage = {}
        for col in df.columns:
            if col.startswith("feature_"):
                feature_name = col.replace("feature_", "")
                feature_usage[feature_name] = int(row[col]) if pd.notna(row[col]) else 0

        account = AccountBehavior(
            account_id=str(row.get("account_id", "")),
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
            feature_usage_count=feature_usage,
            email_open_rate=float(row.get("email_open_rate", 0.0)),
            subscription_age_days=int(row.get("subscription_age_days", 0)),
            plan_level=str(row.get("plan_level", "basic")),
        )
        accounts.append(account)
    return accounts


def load_accounts_from_json(file_path: str) -> List[AccountBehavior]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    accounts = []
    for item in data:
        account = AccountBehavior(**item)
        accounts.append(account)
    return accounts


def load_accounts(file_path: str) -> List[AccountBehavior]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return load_accounts_from_csv(file_path)
    elif suffix in (".json", ".jsonl"):
        return load_accounts_from_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def collect_feature_summary(accounts: List[AccountBehavior]) -> dict:
    if not accounts:
        return {}

    df = pd.DataFrame([a.__dict__ for a in accounts])

    summary = {
        "total_accounts": len(accounts),
        "avg_login_count_30d": float(df["login_count_last_30d"].mean()),
        "avg_last_login_days": float(df["last_login_days_ago"].mean()),
        "avg_transactions_30d": float(df["total_transactions_last_30d"].mean()),
        "avg_transaction_amount": float(df["transaction_amount_last_30d"].mean()),
        "avg_support_tickets": float(df["support_tickets_last_30d"].mean()),
        "avg_payment_failures": float(df["payment_failures_last_30d"].mean()),
        "plan_distribution": df["plan_level"].value_counts().to_dict(),
    }
    return summary
