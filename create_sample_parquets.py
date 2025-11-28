import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("./data")
OUT.mkdir(exist_ok=True, parents=True)

def make_base(seed=0, multiplier=1.0):
    np.random.seed(seed)
    pages = [
        {"page_id": 1, "page": "home"},
        {"page_id": 2, "page": "catalog"},
        {"page_id": 3, "page": "product"},
        {"page_id": 4, "page": "checkout"},
        {"page_id": 5, "page": "faq"},
    ]
    rows = []
    for p in pages:
        sessions = int(np.random.randint(200, 2000) * multiplier)
        clicks = int(sessions * np.random.uniform(1.0, 4.0))
        time_on_page = float(np.random.uniform(8, 120) * multiplier)  # seconds
        bounce_rate = float(np.clip(np.random.normal(0.35, 0.12), 0.02, 0.95))
        exit_rate = float(np.clip(np.random.normal(0.25, 0.1), 0.01, 0.99))
        scroll_25 = float(np.random.uniform(0.5, 0.95))
        scroll_50 = float(np.random.uniform(0.25, scroll_25))
        scroll_75 = float(np.random.uniform(0.1, scroll_50))
        scroll_100 = float(np.random.uniform(0.05, scroll_75))
        pages_per_session = float(np.random.uniform(1.2, 5.0))
        search_usage_pct = float(np.random.uniform(0.02, 0.25))
        conversions = int(sessions * np.random.uniform(0.01, 0.2))
        back_button_clicks = int(np.random.randint(0, int(sessions*0.1)))
        time_to_first_click = float(np.random.uniform(0.5, 6.0))
        form_started = int(np.random.randint(0, int(sessions*0.3)))
        form_abandoned = int(form_started * np.random.uniform(0.05, 0.7))
        field_errors = int(form_started * np.random.uniform(0.0, 0.15))
        time_to_complete_form = float(np.random.uniform(10, 300))
        field_focuses = int(form_started * np.random.uniform(1.0, 6.0))

        funnel_step_1 = int(sessions * np.random.uniform(0.6, 1.0))
        funnel_step_2 = int(funnel_step_1 * np.random.uniform(0.3, 0.95))
        funnel_step_3 = int(funnel_step_2 * np.random.uniform(0.2, 0.95))

        rows.append({
            **p,
            "sessions": sessions,
            "clicks": clicks,
            "time_on_page_avg": time_on_page,
            "bounce_rate": bounce_rate,
            "exit_rate": exit_rate,
            "scroll_25": scroll_25,
            "scroll_50": scroll_50,
            "scroll_75": scroll_75,
            "scroll_100": scroll_100,
            "pages_per_session": pages_per_session,
            "search_usage_pct": search_usage_pct,
            "conversions": conversions,
            "back_button_clicks": back_button_clicks,
            "time_to_first_click": time_to_first_click,
            "form_started": form_started,
            "form_abandoned": form_abandoned,
            "field_errors": field_errors,
            "time_to_complete_form": time_to_complete_form,
            "field_focuses": field_focuses,
            "funnel_step_1": funnel_step_1,
            "funnel_step_2": funnel_step_2,
            "funnel_step_3": funnel_step_3,
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df_a = make_base(seed=42, multiplier=1.0)
    df_a.to_parquet(OUT / "ux_version_A.parquet", index=False)
    print("Saved data to", OUT / "ux_version_A.parquet")

    df_b = make_base(seed=99, multiplier=1.0)

    df_b.loc[df_b.page == "product", "bounce_rate"] *= 1.8
    df_b.loc[df_b.page == "product", "time_on_page_avg"] *= 0.6
    df_b.loc[df_b.page == "product", "clicks"] = (df_b.loc[df_b.page == "product", "clicks"] * 0.7).astype(int)

    df_b.loc[df_b.page == "checkout", "funnel_step_3"] = (df_b.loc[df_b.page == "checkout", "funnel_step_3"] * 0.4).astype(int)
    df_b.loc[df_b.page == "checkout", "conversions"] = (df_b.loc[df_b.page == "checkout", "conversions"] * 0.45).astype(int)

    df_b.loc[df_b.page == "catalog", "pages_per_session"] *= 1.5
    df_b.loc[df_b.page == "catalog", "search_usage_pct"] *= 1.4

    df_b.to_parquet(OUT / "ux_version_B.parquet", index=False)
    print("Saved data to", OUT / "ux_version_B.parquet")
