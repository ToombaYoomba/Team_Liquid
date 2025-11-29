from pathlib import Path
import pandas as pd
import json
import numpy as np

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

def _analyze_hits_data(df: pd.DataFrame) -> dict:
    """Детальный анализ hits данных"""
    analysis = {}
    
    analysis["total_hits"] = len(df)

    if "ym:pv:URL" in df.columns:
        def extract_url_category(url):
            if isinstance(url, str):
                if '/bachelor/' in url:
                    return 'bachelor'
                elif '/master/' in url:
                    return 'master' 
                elif '/base/' in url:
                    return 'base'
                elif '/rating/' in url:
                    return 'rating'
                elif '/results/' in url:
                    return 'results'
                elif '/news/' in url:
                    return 'news'
                elif url.endswith('/'):
                    return 'homepage'
                else:
                    return 'other'
            return 'unknown'
        
        df['url_category'] = df['ym:pv:URL'].apply(extract_url_category)

        category_analysis = df.groupby("url_category").agg({
            'ym:pv:watchID': 'count'
        }).rename(columns={'ym:pv:watchID': 'views'}).reset_index()
        
        category_analysis['percentage'] = (category_analysis['views'] / len(df) * 100).round(2)
        analysis["url_category_analysis"] = category_analysis.to_dict('records')

        url_analysis = df.groupby("ym:pv:URL").agg({
            'ym:pv:watchID': 'count'
        }).rename(columns={'ym:pv:watchID': 'views'}).reset_index()
        
        url_analysis['percentage'] = (url_analysis['views'] / len(df) * 100).round(2)
        analysis["top_urls"] = url_analysis.nlargest(15, 'views').to_dict('records')
        analysis["total_unique_urls"] = len(url_analysis)

    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_cols:
        for time_col in time_cols:
            if pd.api.types.is_datetime64_any_dtype(df[time_col]):
                analysis["time_range"] = {
                    "start": df[time_col].min().isoformat(),
                    "end": df[time_col].max().isoformat(),
                    "total_days": (df[time_col].max() - df[time_col].min()).days
                }

                df['hour'] = df[time_col].dt.hour
                df['day_of_week'] = df[time_col].dt.day_name()
                
                analysis["hourly_distribution"] = df['hour'].value_counts().sort_index().to_dict()
                analysis["daily_distribution"] = df['day_of_week'].value_counts().to_dict()
                break

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        numeric_stats = {}
        for col in numeric_cols:
            if col not in ['ym:pv:watchID', 'hour']:
                numeric_stats[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "non_zero_count": int((df[col] != 0).sum()),
                    "zero_percentage": float((df[col] == 0).sum() / len(df) * 100)
                }
        analysis["numeric_stats"] = numeric_stats
    
    return analysis

def _analyze_visits_data(df: pd.DataFrame) -> dict:
    """Детальный анализ visits данных"""
    analysis = {}
    
    analysis["total_visits"] = len(df)

    if "ym:s:watchIDs" in df.columns:
        df['session_depth'] = df['ym:s:watchIDs'].apply(
            lambda x: len(x) if isinstance(x, list) else 1
        )
        
        session_metrics = {
            "avg_session_depth": float(df['session_depth'].mean()),
            "median_session_depth": float(df['session_depth'].median()),
            "std_session_depth": float(df['session_depth'].std()),
            "max_session_depth": int(df['session_depth'].max()),
            "min_session_depth": int(df['session_depth'].min()),
            "single_page_sessions": int((df['session_depth'] == 1).sum()),
            "multi_page_sessions": int((df['session_depth'] > 1).sum()),
            "bounce_rate": float((df['session_depth'] == 1).sum() / len(df) * 100),
            "depth_distribution": df['session_depth'].value_counts().sort_index().to_dict()
        }

        session_metrics["depth_percentiles"] = {
            "q10": float(df['session_depth'].quantile(0.1)),
            "q25": float(df['session_depth'].quantile(0.25)),
            "q50": float(df['session_depth'].quantile(0.5)),
            "q75": float(df['session_depth'].quantile(0.75)),
            "q90": float(df['session_depth'].quantile(0.9))
        }
        
        analysis["session_metrics"] = session_metrics

    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        for date_col in date_cols:
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                analysis["date_range"] = {
                    "start": df[date_col].min().isoformat(),
                    "end": df[date_col].max().isoformat(),
                    "total_days": (df[date_col].max() - df[date_col].min()).days
                }

                daily_visits = df[date_col].dt.date.value_counts().sort_index()
                analysis["visits_by_date"] = {
                    "dates": [d.isoformat() for d in daily_visits.index],
                    "counts": daily_visits.values.tolist(),
                    "avg_visits_per_day": float(daily_visits.mean()),
                    "max_visits_day": {
                        "date": daily_visits.idxmax().isoformat(),
                        "count": int(daily_visits.max())
                    }
                }
                break
    
    return analysis

def _load_combined_ux_data(hits_file: str, visits_file: str) -> str:
    """
    Объединяет данные hits и visits для комплексного UX анализа
    """
    hits_df = _read_parquet_df(hits_file)
    visits_df = _read_parquet_df(visits_file)

    hits_analysis = _analyze_hits_data(hits_df)
    visits_analysis = _analyze_visits_data(visits_df)

    combined_metrics = {
        "hits_per_visit": hits_analysis["total_hits"] / visits_analysis["total_visits"] if visits_analysis["total_visits"] > 0 else 0,
        "total_data_points": hits_analysis["total_hits"] + visits_analysis["total_visits"]
    }

    combined_summary = {
        "hits_analysis": hits_analysis,
        "visits_analysis": visits_analysis,
        "combined_metrics": combined_metrics,
        "data_sources": {
            "hits_file": hits_file,
            "visits_file": visits_file,
            "hits_columns": hits_df.columns.tolist(),
            "visits_columns": visits_df.columns.tolist()
        }
    }
    
    return json.dumps(combined_summary, ensure_ascii=False, indent=2)

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
    
    @mcp.tool()
    def load_combined_ux_analysis(hits_file: str, visits_file: str) -> str:
        return _load_combined_ux_data(hits_file, visits_file)

except ImportError:
    pass