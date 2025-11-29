import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("./data")


def _read_parquet_df(path: str) -> pd.DataFrame:
    """
    Загружает DataFrame из parquet файла.
    Ищет ТОЛЬКО в папке ./data
    """
    if path.startswith("data/"):
        p = Path(path)
    else:
        p = DATA_DIR / path

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_parquet(p)


def _analyze_full_metrics_data(df: pd.DataFrame) -> dict:
    """
    Анализ данных из full_metrics.parquet
    Предполагается структура: 2 строки (версии), много столбцов с метриками
    """
    analysis = {}

    analysis["data_structure"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "versions_found": df.iloc[:, 0].tolist() if len(df) > 0 else []
    }
    
    if len(df) == 2:
        v1_data = df.iloc[0]
        v2_data = df.iloc[1]

        metrics_comparison = {}
        
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                v1_val = v1_data[column]
                v2_val = v2_data[column]

                if v1_val != 0:
                    change_percent = ((v2_val - v1_val) / abs(v1_val)) * 100
                else:
                    change_percent = 100 if v2_val != 0 else 0
                
                metrics_comparison[column] = {
                    "v1_value": float(v1_val),
                    "v2_value": float(v2_val),
                    "change_percent": float(change_percent)
                }
        
        analysis["metrics_comparison"] = metrics_comparison
        analysis["total_metrics"] = len(metrics_comparison)

        significant_changes = {
            col: data for col, data in metrics_comparison.items() 
            if abs(data["change_percent"]) > 20
        }
        analysis["significant_changes_20pct"] = significant_changes
        analysis["significant_changes_count"] = len(significant_changes)
    
    return analysis


def _load_full_metrics_analysis(metrics_file: str) -> str:
    """
    Загрузка и анализ данных из full_metrics.parquet
    """
    df = _read_parquet_df(metrics_file)
    
    analysis = _analyze_full_metrics_data(df)

    analysis["data_info"] = {
        "source_file": metrics_file,
        "columns_available": df.columns.tolist(),
        "data_types": {col: str(df[col].dtype) for col in df.columns}
    }
    
    return json.dumps(analysis, ensure_ascii=False, indent=2)


def _load_parquet_summary_direct(filename: str) -> str:
    """
    Прямая функция для преобразования parquet в JSON-сводку
    """
    df = _read_parquet_df(filename)
    return df.to_json(orient="records", force_ascii=False)


try:
    from fastmcp import FastMCP

    mcp = FastMCP("UX MCP Tools")

    @mcp.tool()
    def load_parquet_as_summary(filename: str) -> str:
        return _load_parquet_summary_direct(filename)

    @mcp.tool()
    def load_full_metrics_analysis(metrics_file: str) -> str:
        return _load_full_metrics_analysis(metrics_file)

except ImportError:
    pass