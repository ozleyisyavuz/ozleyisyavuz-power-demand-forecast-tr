from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor


DATA_PATH = Path("data/processed/renewables.csv")
MODEL_DIR = Path("models")


def _build_model(alpha: float) -> Pipeline:
    cat_cols = ["hour", "dayofweek", "month", "is_weekend"]
    num_cols = ["wind_speed_mps", "ghi_wm2", "temperature_c"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )

    gbr = GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    return Pipeline([("pre", pre), ("model", gbr)])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Veri yok. Önce üret: python -m renewable_generation_forecast.data.make_dataset")

    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    X = pd.DataFrame(
        {
            "hour": df["timestamp"].dt.hour,
            "dayofweek": df["timestamp"].dt.dayofweek,
            "month": df["timestamp"].dt.month,
            "is_weekend": (df["timestamp"].dt.dayofweek >= 5).astype(int),
            "wind_speed_mps": df["wind_speed_mps"],
            "ghi_wm2": df["ghi_wm2"],
            "temperature_c": df["temperature_c"],
        }
    )
    y = df["generation_mw"].astype(float)

    # Time-ordered holdout: last 7 days
    holdout = 7 * 24
    X_train, X_val = X.iloc[:-holdout], X.iloc[-holdout:]
    y_train, y_val = y.iloc[:-holdout], y.iloc[-holdout:]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    models = {}
    for q in (0.10, 0.50, 0.90):
        m = _build_model(alpha=q)
        m.fit(X_train, y_train)
        models[q] = m
        joblib.dump(m, MODEL_DIR / f"q{int(q*100):02d}.joblib")

    p10 = models[0.10].predict(X_val)
    p50 = models[0.50].predict(X_val)
    p90 = models[0.90].predict(X_val)

    mae = float((y_val - p50).abs().mean())
    mape = float(((y_val - p50).abs() / (y_val.clip(lower=1.0))).mean() * 100)
    coverage = float(((y_val >= p10) & (y_val <= p90)).mean() * 100)

    print(f"Validation MAE (MW): {mae:.2f}")
    print(f"Validation MAPE (%): {mape:.2f}")
    print(f"Prediction Interval Coverage P10–P90 (%): {coverage:.2f}")
    print(f"Saved models to: {MODEL_DIR.resolve()}")


if __name__ == "__main__":
    main()
