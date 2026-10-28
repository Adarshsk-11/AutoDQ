"""Simple in-memory dataset store.

Per spec section 14: 'If a database is unnecessary for the MVP, DO NOT
introduce one.' This process-local dict is sufficient for a single-user
demo/MVP and keeps the architecture free to add persistence later without
touching the analysis engine.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.services.ingestion import ColumnSchema


@dataclass
class StoredDataset:
    dataset_id: str
    filename: str
    df: pd.DataFrame
    schema: list[ColumnSchema]
    ingestion_warnings: list[str]
    last_analysis: dict[str, Any] | None = None
    last_target_column: str | None = None
    last_domain: str = "general"


_STORE: dict[str, StoredDataset] = {}


def save_dataset(filename: str, df: pd.DataFrame, schema: list[ColumnSchema], warnings: list[str]) -> StoredDataset:
    dataset_id = uuid.uuid4().hex[:12]
    record = StoredDataset(
        dataset_id=dataset_id,
        filename=filename,
        df=df,
        schema=schema,
        ingestion_warnings=warnings,
    )
    _STORE[dataset_id] = record
    return record


def get_dataset(dataset_id: str) -> StoredDataset | None:
    return _STORE.get(dataset_id)


def all_datasets() -> list[StoredDataset]:
    return list(_STORE.values())
