from pathlib import Path
import pandas as pd
import json

DATA_DIR = Path("./data")


def _read_parquet_df(path: str) -> pd.DataFrame:
    """
    Загружает DataFrame из parquet файла.
    Ищет сначала в текущей папке, потом в ./data
    """
    p = Path(path)
    if not p.exists():
        p = DATA_DIR / path
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_parquet(p)


def _load_parquet_summary_direct(filename: str) -> str:
    """
    Прямая функция для преобразования parquet в JSON-сводку
    """
    df = _read_parquet_df(filename)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if "page" in df.columns:
        summary = df.groupby("page")[numeric].mean().reset_index()
    else:
        summary = df[numeric]
    return summary.to_json(orient="records", force_ascii=False)

try:
    from fastmcp import FastMCP
    mcp = FastMCP("UX MCP Tools")

    @mcp.tool()
    def load_parquet_as_summary(filename: str) -> str:
        return _load_parquet_summary_direct(filename)

except ImportError:
    pass
