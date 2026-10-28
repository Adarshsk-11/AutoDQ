from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from app.schemas.api_schemas import AnalyzeRequest
from app.services.analysis_service import analysis_result_to_dict, run_full_analysis
from app.services.auto_fix import apply_fixes
from app.services.ingestion import IngestionError, detect_schema, load_dataset
from app.services.report_export import build_html_report
from app.utils.store import all_datasets, get_dataset, save_dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("")
def list_datasets():
    """Session history: every dataset uploaded so far, with whether it's
    been analyzed yet. Useful for re-visiting a previous upload without
    re-selecting the file, and as a foundation for comparing runs."""
    return [
        {
            "dataset_id": r.dataset_id,
            "filename": r.filename,
            "n_rows": int(r.df.shape[0]),
            "n_columns": int(r.df.shape[1]),
            "analyzed": r.last_analysis is not None,
            "last_target_column": r.last_target_column,
            "last_domain": r.last_domain,
            "dqi_0_100": r.last_analysis["dqi"]["dqi_0_100"] if r.last_analysis else None,
        }
        for r in all_datasets()
    ]


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        result = load_dataset(contents, file.filename or "dataset")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    record = save_dataset(result.source_filename, result.df, result.schema, result.warnings)

    return {
        "dataset_id": record.dataset_id,
        "filename": record.filename,
        "n_rows": int(result.df.shape[0]),
        "n_columns": int(result.df.shape[1]),
        "columns": [
            {"name": s.name, "dtype": s.dtype, "n_unique": s.n_unique, "n_missing": s.n_missing}
            for s in result.schema
        ],
        "warnings": result.warnings,
    }


@router.get("/{dataset_id}")
def get_dataset_info(dataset_id: str):
    record = get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {
        "dataset_id": record.dataset_id,
        "filename": record.filename,
        "n_rows": int(record.df.shape[0]),
        "n_columns": int(record.df.shape[1]),
        "columns": [
            {"name": s.name, "dtype": s.dtype, "n_unique": s.n_unique, "n_missing": s.n_missing}
            for s in record.schema
        ],
        "warnings": record.ingestion_warnings,
        "last_target_column": record.last_target_column,
        "last_domain": record.last_domain,
        "analyzed": record.last_analysis is not None,
    }


@router.post("/{dataset_id}/analyze")
def analyze_dataset(dataset_id: str, request: AnalyzeRequest):
    record = get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    result = run_full_analysis(
        record.df,
        record.schema,
        target_column=request.target_column,
        domain=request.domain,
        ingestion_warnings=record.ingestion_warnings,
    )
    result_dict = analysis_result_to_dict(result)
    record.last_analysis = result_dict
    record.last_target_column = request.target_column
    record.last_domain = request.domain
    return result_dict


def _require_analysis(dataset_id: str) -> dict:
    record = get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not record.last_analysis:
        raise HTTPException(status_code=409, detail="Dataset has not been analyzed yet. Call /analyze first.")
    return record.last_analysis


@router.get("/{dataset_id}/profile")
def get_profile(dataset_id: str):
    return _require_analysis(dataset_id)["profile"]


@router.get("/{dataset_id}/metrics")
def get_metrics(dataset_id: str):
    analysis = _require_analysis(dataset_id)
    return {
        "dqi": analysis["dqi"],
        "completeness": analysis["completeness"],
        "consistency": analysis["consistency"],
        "validity": analysis["validity"],
        "uniqueness": analysis["uniqueness"],
        "balance": analysis["balance"],
        "informativeness": analysis["informativeness"],
    }


@router.get("/{dataset_id}/informativeness")
def get_informativeness(dataset_id: str):
    """Experimental diagnostic, separate from DQI -- see
    app.analyzers.informativeness for why it isn't folded into /metrics'
    dqi field."""
    return _require_analysis(dataset_id)["informativeness"]


@router.get("/{dataset_id}/issues")
def get_issues(dataset_id: str):
    return _require_analysis(dataset_id)["issues"]


@router.get("/{dataset_id}/recommendations")
def get_recommendations(dataset_id: str):
    return _require_analysis(dataset_id)["recommendations"]


@router.get("/{dataset_id}/model-performance")
def get_model_performance(dataset_id: str):
    return _require_analysis(dataset_id)["baseline"]


@router.get("/{dataset_id}/report")
def get_report(dataset_id: str, format: str = Query("json", pattern="^(json|html)$")):
    record = get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not record.last_analysis:
        raise HTTPException(status_code=409, detail="Dataset has not been analyzed yet. Call /analyze first.")

    dataset_meta = {
        "filename": record.filename,
        "n_rows": int(record.df.shape[0]),
        "n_columns": int(record.df.shape[1]),
    }

    if format == "html":
        html_content = build_html_report(record.last_analysis, dataset_meta)
        return HTMLResponse(content=html_content)

    return JSONResponse(content={"dataset": dataset_meta, "analysis": record.last_analysis})


def _derive_cleaned_filename(original: str) -> str:
    if "." in original:
        stem, ext = original.rsplit(".", 1)
        return f"{stem}_cleaned.{ext}"
    return f"{original}_cleaned"


@router.post("/{dataset_id}/clean")
def clean_dataset(dataset_id: str):
    """Applies the auto-fix engine (see app.services.auto_fix) targeting
    exactly the five DQI dimensions -- Completeness, Consistency, Validity,
    Uniqueness, Balance -- to this dataset, saves the result as a NEW
    dataset (never mutates the original in place), re-runs the full
    analysis on it, and returns a before/after comparison plus a
    human-readable log of every fix applied. The cleaned dataset gets its
    own dataset_id and can be fetched/downloaded/re-analyzed like any other
    upload via GET /{cleaned_dataset_id}/download.
    """
    record = get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    if not record.last_analysis:
        raise HTTPException(status_code=409, detail="Dataset has not been analyzed yet. Call /analyze first.")

    column_types = record.last_analysis["profile"]["column_types"]
    target_column = record.last_target_column
    domain = record.last_domain

    fix_result = apply_fixes(record.df, column_types, target_column)
    cleaned_schema = detect_schema(fix_result.cleaned_df)

    after_result = run_full_analysis(
        fix_result.cleaned_df, cleaned_schema, target_column=target_column, domain=domain,
    )
    after_dict = analysis_result_to_dict(after_result)

    cleaned_record = save_dataset(
        _derive_cleaned_filename(record.filename), fix_result.cleaned_df, cleaned_schema, []
    )
    cleaned_record.last_analysis = after_dict
    cleaned_record.last_target_column = target_column
    cleaned_record.last_domain = domain

    return {
        "cleaned_dataset_id": cleaned_record.dataset_id,
        "cleaned_filename": cleaned_record.filename,
        "fix_log": fix_result.fix_log,
        "before": {
            "n_rows": int(record.df.shape[0]),
            "dqi": record.last_analysis["dqi"],
        },
        "after": {
            "n_rows": int(fix_result.cleaned_df.shape[0]),
            "dqi": after_dict["dqi"],
        },
        "download_url": f"/api/datasets/{cleaned_record.dataset_id}/download",
    }


@router.get("/{dataset_id}/download")
def download_dataset(dataset_id: str):
    """Streams the dataset's current data as a CSV attachment. Works for
    both original uploads and cleaned datasets produced by POST /clean --
    both are stored the same way, so this endpoint doesn't need to know
    which kind it's serving."""
    record = get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    csv_bytes = record.df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{record.filename}"'},
    )
