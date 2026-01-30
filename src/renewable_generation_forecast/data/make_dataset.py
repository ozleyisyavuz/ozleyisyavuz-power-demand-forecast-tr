from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _solar_shape(hour: int) -> float:
    
    if hour < 6 or hour > 18:
        return 0.0
    x = (hour - 6) / 12.0  # 0..1
    return float(np.sin(np.pi * x))


def make_synthetic_renewables(days: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start="2024-01-01", periods=days * 24, freq="h")

    df = pd.DataFrame({"timestamp": idx})
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    
    seasonal = np.sin(2 * np.pi * (df.index / (24 * 365)))  
    daily = np.sin(2 * np.pi * (df["hour"] / 24))          

    
    wind_noise = rng.normal(0, 1.2, size=len(df))
    df["wind_speed_mps"] = (6.5 + 1.8 * seasonal + 1.0 * daily + wind_noise).clip(lower=0.0)

   
    temp_noise = rng.normal(0, 1.0, size=len(df))
    df["temperature_c"] = (16 + 8 * seasonal + 2.0 * daily + temp_noise)

    clouds = rng.uniform(0.1, 0.8, size=len(df))  
    solar_shape = df["hour"].apply(_solar_shape).to_numpy()
    ghi_base = 800 * solar_shape * (0.55 + 0.45 * (1 + seasonal) / 2)  
    df["ghi_wm2"] = (ghi_base * (1 - clouds)).clip(lower=0.0)

    
    installed_wind_mw = 10000.0
    installed_solar_mw = 8000.0

    
    wind_norm = (df["wind_speed_mps"] / 12.0).clip(0, 1) ** 3
    wind_mw = installed_wind_mw * wind_norm

    
    solar_norm = (df["ghi_wm2"] / 900.0).clip(0, 1)
    solar_mw = installed_solar_mw * solar_norm

    
    noise = rng.normal(0, 250, size=len(df))
    weekend_effect = -100 * df["is_weekend"]
    df["wind_mw"] = (wind_mw + noise * 0.4).clip(lower=0.0)
    df["solar_mw"] = (solar_mw + noise * 0.3).clip(lower=0.0)
    df["generation_mw"] = (df["wind_mw"] + df["solar_mw"] + weekend_effect + noise * 0.2).clip(lower=0.0)

    return df[["timestamp", "wind_speed_mps", "ghi_wm2", "temperature_c", "generation_mw"]]


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "renewables.csv"
    df = make_synthetic_renewables(days=120, seed=42)
    df.to_csv(out_path, index=False)
    print(f"Saved dataset: {out_path} shape={df.shape}")


if __name__ == "__main__":
    main()
