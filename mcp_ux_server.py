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


def _read_excel_file(path: str) -> str:
    """
    Загружает Excel файл и преобразует в JSON строку
    """
    if path.startswith("data/"):
        p = Path(path)
    else:
        p = DATA_DIR / path

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    excel_data = {}
    xl = pd.ExcelFile(p)
    
    for sheet_name in xl.sheet_names:
        df = pd.read_excel(p, sheet_name=sheet_name)
        excel_data[sheet_name] = {
            "columns": df.columns.tolist(),
            "data": df.to_dict('records')
        }
    
    return json.dumps(excel_data, ensure_ascii=False, indent=2)


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


def _analyze_goals_data(df: pd.DataFrame) -> dict:
    """
    Анализ данных по целям из goal_stats_common_v1_v2.parquet
    Структура: goal_id, avg_steps, avg_duration_sec, version
    8 строк: 4 цели для v1, 4 цели для v2
    """
    analysis = {}
    
    analysis["data_structure"] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns.tolist()
    }

    required_columns = ['goal_id', 'avg_steps', 'avg_duration_sec', 'version']
    if not all(col in df.columns for col in required_columns):
        analysis["error"] = f"Missing required columns. Found: {df.columns.tolist()}, Required: {required_columns}"
        return analysis

    v1_data = df[df['version'] == 'v1']
    v2_data = df[df['version'] == 'v2']
    
    analysis["versions_info"] = {
        "v1_goals_count": len(v1_data),
        "v2_goals_count": len(v2_data),
        "v1_goal_ids": v1_data['goal_id'].tolist(),
        "v2_goal_ids": v2_data['goal_id'].tolist()
    }

    goals_comparison = {}
    
    for goal_id in v1_data['goal_id'].unique():
        v1_goal = v1_data[v1_data['goal_id'] == goal_id]
        v2_goal = v2_data[v2_data['goal_id'] == goal_id]
        
        if len(v1_goal) > 0 and len(v2_goal) > 0:
            v1_row = v1_goal.iloc[0]
            v2_row = v2_goal.iloc[0]
            
            goal_comparison = {}

            v1_steps = v1_row['avg_steps']
            v2_steps = v2_row['avg_steps']
            steps_change = ((v2_steps - v1_steps) / v1_steps * 100) if v1_steps != 0 else 100
            
            goal_comparison['avg_steps'] = {
                "v1_value": float(v1_steps),
                "v2_value": float(v2_steps),
                "change_percent": float(steps_change)
            }

            v1_duration = v1_row['avg_duration_sec']
            v2_duration = v2_row['avg_duration_sec']
            duration_change = ((v2_duration - v1_duration) / v1_duration * 100) if v1_duration != 0 else 100
            
            goal_comparison['avg_duration_sec'] = {
                "v1_value": float(v1_duration),
                "v2_value": float(v2_duration),
                "change_percent": float(duration_change)
            }
            
            goals_comparison[str(goal_id)] = goal_comparison
    
    analysis["goals_comparison"] = goals_comparison
    analysis["total_goals"] = len(goals_comparison)

    significant_goals = {}
    for goal_id, metrics in goals_comparison.items():
        significant_metrics = {}
        for metric_name, metric_data in metrics.items():
            if abs(metric_data["change_percent"]) > 20:
                significant_metrics[metric_name] = metric_data
        
        if significant_metrics:
            significant_goals[goal_id] = significant_metrics
    
    analysis["significant_goals"] = significant_goals
    analysis["significant_goals_count"] = len(significant_goals)
    
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


def _load_goals_analysis(goals_file: str) -> str:
    """
    Загрузка и анализ данных по целям
    """
    df = _read_parquet_df(goals_file)
    
    analysis = _analyze_goals_data(df)

    analysis["data_info"] = {
        "source_file": goals_file,
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

    @mcp.tool()
    def load_goals_analysis(goals_file: str) -> str:
        return _load_goals_analysis(goals_file)

    @mcp.tool()
    def load_excel_file(excel_file: str) -> str:
        return _read_excel_file(excel_file)

except ImportError:
    pass