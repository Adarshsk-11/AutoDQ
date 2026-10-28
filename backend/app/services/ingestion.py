"""
Dataset ingestion: loading, validation, and schema detection.

Supports CSV, Excel (.xlsx/.xls), and JSON. Never raises on "bad but readable"
data -- it degrades gracefully and reports warnings instead, per the
error-handling requirements (section 15 of the spec).
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Configurable via MAX_FILE_SIZE_MB in .env; defaults to 50MB, generous for an MVP.
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "50")) * 1024 * 1024
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}


class IngestionError(Exception):
    """Raised only for truly unrecoverable input (empty file, bad extension,
    size limit exceeded, completely unparsable content)."""


@dataclass
class ColumnSchema:
    name: str
    dtype: str  # "numeric" | "categorical" | "datetime" | "boolean" | "text" | "unknown"
    pandas_dtype: str
    n_unique: int
    n_missing: int


@dataclass
class IngestResult:
    df: pd.DataFrame
    schema: list[ColumnSchema]
    warnings: list[str] = field(default_factory=list)
    source_filename: str = ""


def _sanitize_filename(filename: str) -> str:
    """Strip path components and suspicious characters. We never use the raw
    filename to build a filesystem path -- datasets are kept in memory /
    behind a generated id -- but we still sanitize before echoing it back
    to the client."""
    name = filename.replace("\\", "/").split("/")[-1]
    return "".join(c for c in name if c.isalnum() or c in "._- ")[:255] or "dataset"


def load_dataset(file_bytes: bytes, filename: str) -> IngestResult:
    """Load a dataset from raw bytes based on its extension.

    Raises IngestionError only for unrecoverable problems. Anything else
    (weird JSON shape, mixed types, etc.) is handled gracefully with a
    warning attached to the result.
    """
    if not file_bytes:
        raise IngestionError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise IngestionError(
            f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024*1024)}MB size limit for this MVP."
        )

    safe_name = _sanitize_filename(filename)
    ext = "." + safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""

    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext or 'unknown'}'. Supported types: CSV, XLSX, JSON."
        )

    warnings: list[str] = []

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        elif ext == ".json":
            df, json_warnings = _load_json(file_bytes)
            warnings.extend(json_warnings)
        else:  # pragma: no cover - guarded above
            raise IngestionError("Unsupported file type.")
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Could not parse the file as {ext.upper()}. It may be corrupted "
            f"or malformed. ({exc.__class__.__name__})"
        ) from exc

    if df is None or df.shape[1] == 0:
        raise IngestionError("The dataset has no columns after parsing.")

    if df.shape[0] == 0:
        warnings.append(
            "The dataset has 0 rows. Most analyses will be unavailable."
        )

    # Drop fully-empty "Unnamed" index columns often produced by bad exports
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        warnings.append(
            f"Dropped {len(unnamed_cols)} empty unnamed column(s), likely stray index columns from export."
        )

    # De-duplicate column names rather than crashing downstream
    if df.columns.duplicated().any():
        df.columns = _dedupe_columns(list(df.columns))
        warnings.append("Duplicate column names were detected and auto-renamed.")

    df.columns = [str(c) for c in df.columns]

    schema = detect_schema(df)

    return IngestResult(df=df, schema=schema, warnings=warnings, source_filename=safe_name)


def _dedupe_columns(cols: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def _load_json(file_bytes: bytes) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    try:
        data: Any = json.loads(file_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        data = json.loads(file_bytes.decode("latin-1"))
        warnings.append("File was not valid UTF-8; decoded as Latin-1.")

    if isinstance(data, list):
        df = pd.json_normalize(data)
    elif isinstance(data, dict):
        # Common shapes: {"data": [...]}, {"records": [...]}, or a single record
        for key in ("data", "records", "rows", "items"):
            if key in data and isinstance(data[key], list):
                df = pd.json_normalize(data[key])
                warnings.append(f"Extracted records from top-level '{key}' key.")
                break
        else:
            if all(not isinstance(v, (dict, list)) for v in data.values()):
                df = pd.json_normalize([data])
                warnings.append("JSON was a single object; treated as a one-row dataset.")
            else:
                # Dict-of-columns shape: {"col1": [...], "col2": [...]}
                try:
                    df = pd.DataFrame(data)
                    warnings.append("Interpreted JSON as a dict-of-columns structure.")
                except Exception as exc:
                    raise IngestionError(
                        "Unrecognized JSON structure. Expected a list of records, "
                        "a dict-of-columns, or a single object."
                    ) from exc
    else:
        raise IngestionError("JSON root must be an object or a list of records.")

    return df, warnings


def _infer_column_type(series: pd.Series) -> str:
    """Best-effort classification into numeric / categorical / datetime / boolean / text."""
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Try datetime parsing on strings, but only trust it if most values parse
    if non_null.dtype == object:
        sample = non_null.sample(min(50, len(non_null)), random_state=0)
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().mean() > 0.9 and sample.astype(str).str.match(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
        ).mean() > 0.5:
            return "datetime"

    n_unique = non_null.nunique()
    n_total = len(non_null)

    if set(non_null.astype(str).str.lower().unique()) <= {"true", "false", "yes", "no", "0", "1"} and n_unique <= 2:
        return "boolean"

    # Heuristic: low-cardinality relative to row count => categorical, else free text
    if n_unique <= max(20, int(0.05 * n_total)):
        return "categorical"

    avg_len = non_null.astype(str).str.len().mean()
    if avg_len > 30:
        return "text"

    return "categorical"


def detect_schema(df: pd.DataFrame) -> list[ColumnSchema]:
    schema = []
    for col in df.columns:
        series = df[col]
        schema.append(
            ColumnSchema(
                name=col,
                dtype=_infer_column_type(series),
                pandas_dtype=str(series.dtype),
                n_unique=int(series.nunique(dropna=True)),
                n_missing=int(series.isna().sum()),
            )
        )
    return schema
