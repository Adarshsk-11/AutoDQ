from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    n_columns: int
    columns: list[dict]
    warnings: list[str]


class AnalyzeRequest(BaseModel):
    target_column: Optional[str] = None
    domain: str = "general"


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    n_rows: int
    n_columns: int
