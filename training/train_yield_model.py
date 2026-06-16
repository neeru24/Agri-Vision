"""
train_yield_model.py
====================
Trains crop-specific XGBoost yield regression models for Agri-Vision.

One model per crop:
  - cotton_yield_model.pkl
  - tomato_yield_model.pkl
  - potato_yield_model.pkl

Saved to: ai_models/

Usage:
    python training/train_yield_model.py
    python training/train_yield_model.py --crop cotton   # single crop only
    python training/train_yield_model.py --no-plots      # skip matplotlib output

Why synthetic data:
    No real labeled yield dataset exists for this project.
    Synthetic data encodes agronomic domain knowledge (ICAR guidelines)
    as a realistic distribution — this is functionally the same as the
    existing hardcoded multipliers, but XGBoost additionally learns
    feature interaction effects (e.g. disease × heat stress) that
    simple multiplicative rules cannot express.
"""

import os
import json
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # Agri-Vision/
CONFIG_PATH = ROOT / "training" / "yield_model_config.json"
MODELS_DIR = ROOT / "ai_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load config ───────────────────────────────────────────────────────────────

with open(CONFIG_PATH) as f:
    CFG = json.load(f)

WEATHER_CFG = CFG["weather_stress"]
TRAIN_CFG = CFG["training"]
XGB_PARAMS = TRAIN_CFG["xgb_params"]


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def _weather_penalty(temp, humidity, precip) -> float:
    """
    Calculate combined weather yield penalty (0.0 to 1.0, where 1.0 = no penalty).
    Encodes nonlinear stress interactions from ICAR agro-advisory guidelines.
    """
    mult = 1.0
    w = WEATHER_CFG

    # Temperature stress — nonlinear above heat threshold
    if temp > w["temperature"]["heat_threshold"]:
        excess = temp - w["temperature"]["heat_threshold"]
        mult *= max(0.50, 1.0 - excess * w["temperature"]["heat_stress_per_degree"])
    elif temp < w["temperature"]["cold_threshold"]:
        deficit = w["temperature"]["cold_threshold"] - temp
        mult *= max(0.60, 1.0 - deficit * w["temperature"]["cold_stress_per_degree"])

    # Humidity stress — disease pressure on bolls/fruit
    if humidity > w["humidity"]["high_threshold"]:
        mult *= w["humidity"]["high_stress_factor"]

    # Heavy rain — boll rot / waterlogging risk
    if precip > w["precipitation"]["heavy_rain_threshold_mm"]:
        mult *= w["precipitation"]["heavy_rain_factor"]

    return round(mult, 4)


def generate_crop_data(crop_name: str, n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate realistic synthetic yield training data for one crop.

    Key design decisions:
    - Disease severity × disease confidence are multiplied → high confidence
      Bacterial Blight (severity 0.45) hurts more than low confidence.
    - Stage fraction × disease effect gives the base health state.
    - Weather penalty is applied on top — interaction effects emerge naturally
      because all three factors combine multiplicatively before noise is added.
    - Gaussian noise (std=5%) prevents the model from memorising the exact
      formula and encourages learning smooth response surfaces.
    """
    crop_cfg = CFG["crops"][crop_name]
    stages = list(crop_cfg["stage_base_fractions"].keys())
    stage_fracs = crop_cfg["stage_base_fractions"]
    diseases = list(crop_cfg["diseases"].keys())
    disease_severity = {d: v["severity"] for d, v in crop_cfg["diseases"].items()}

    base_min = crop_cfg["base_yield_min_kg_acre"]
    base_max = crop_cfg["base_yield_max_kg_acre"]
    base_mean = crop_cfg["healthy_yield_mean"]

    rows = []
    for _ in range(n_samples):
        # ── Sample inputs ──────────────────────────────────────────────────
        stage = rng.choice(stages)
        disease = rng.choice(diseases)
        disease_conf = rng.uniform(0.40, 0.99)   # ResNet50 confidence
        field_acres = rng.uniform(0.5, 25.0)

        # Weather — sampled from realistic Indian agricultural ranges
        temp = rng.uniform(10, 45)
        humidity = rng.uniform(25, 95)
        precip = rng.choice(
            [0.0, rng.uniform(0, 3), rng.uniform(3, 15)],
            p=[0.55, 0.30, 0.15]
        )

        # ── Compute yield ──────────────────────────────────────────────────
        stage_frac = stage_fracs[stage]
        sev = disease_severity[disease]

        # Effective disease impact = severity scaled by model confidence
        # If conf=0.40 and severity=0.45 → actual impact = 0.18 (mild)
        # If conf=0.95 and severity=0.45 → actual impact = 0.43 (severe)
        effective_disease_loss = sev * disease_conf

        # Health factor after disease
        health_factor = max(0.10, 1.0 - effective_disease_loss)

        # Weather penalty (nonlinear)
        weather_factor = _weather_penalty(temp, humidity, precip)

        # Combined factor
        combined = stage_frac * health_factor * weather_factor

        # Base yield — sample from crop range, biased toward mean
        base_yield = rng.normal(loc=base_mean, scale=(base_max - base_min) / 6)
        base_yield = np.clip(base_yield, base_min, base_max)

        # Final yield with Gaussian noise (±5% realistic field variation)
        noise = rng.normal(1.0, TRAIN_CFG["noise_std"])
        yield_kg_acre = round(float(base_yield * combined * noise), 2)
        yield_kg_acre = max(10.0, yield_kg_acre)   # floor: never negative

        rows.append({
            "crop_type":          crop_name,
            "growth_stage":       stage,
            "disease_class":      disease,
            "disease_confidence": round(float(disease_conf), 4),
            "temperature":        round(float(temp), 1),
            "humidity":           round(float(humidity), 1),
            "precipitation":      round(float(precip), 2),
            "field_acres":        round(float(field_acres), 2),
            "yield_kg_acre":      yield_kg_acre,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, stage_enc=None, disease_enc=None):
    """
    Encode categoricals + add interaction features.
    Encoders are fit on first call (training), reused on subsequent calls (inference).
    Returns (X, stage_encoder, disease_encoder).
    """
    df = df.copy()

    # Label encode growth stage
    if stage_enc is None:
        stage_enc = LabelEncoder()
        df["stage_enc"] = stage_enc.fit_transform(df["growth_stage"])
    else:
        # Handle unseen labels at inference time gracefully
        known = set(stage_enc.classes_)
        df["growth_stage"] = df["growth_stage"].apply(
            lambda x: x if x in known else stage_enc.classes_[0]
        )
        df["stage_enc"] = stage_enc.transform(df["growth_stage"])

    # Label encode disease class
    if disease_enc is None:
        disease_enc = LabelEncoder()
        df["disease_enc"] = disease_enc.fit_transform(df["disease_class"])
    else:
        known = set(disease_enc.classes_)
        df["disease_class"] = df["disease_class"].apply(
            lambda x: x if x in known else disease_enc.classes_[0]
        )
        df["disease_enc"] = disease_enc.transform(df["disease_class"])

    # Interaction features — these are the main reason XGBoost > multipliers
    # The model learns the exact shape of these interactions from data
    df["disease_impact"] = df["disease_confidence"] * df["disease_enc"]
    df["temp_humidity_stress"] = (
        (df["temperature"] - 30).abs() * (df["humidity"] / 100)
    )
    df["stage_health_interact"] = df["stage_enc"] * (1 - df["disease_confidence"])

    feature_cols = [
        "stage_enc",
        "disease_enc",
        "disease_confidence",
        "temperature",
        "humidity",
        "precipitation",
        "field_acres",
        "disease_impact",
        "temp_humidity_stress",
        "stage_health_interact",
    ]

    X = df[feature_cols].values
    return X, stage_enc, disease_enc, feature_cols


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_crop_model(crop_name: str, show_plots: bool = True) -> dict:
    """
    Full training pipeline for one crop.
    Returns metadata dict with CV scores and feature importances.
    """
    log.info(f"{'='*60}")
    log.info(f"Training yield model: {crop_name.upper()}")
    log.info(f"{'='*60}")

    # ── Generate data ──
    rng = np.random.default_rng(seed=42)
    n = TRAIN_CFG["samples_per_crop"]
    log.info(f"Generating {n} synthetic samples...")
    df = generate_crop_data(crop_name, n, rng)
    log.info(f"Yield range: {df['yield_kg_acre'].min():.1f} – {df['yield_kg_acre'].max():.1f} kg/acre")
    log.info(f"Yield mean:  {df['yield_kg_acre'].mean():.1f} kg/acre")

    # ── Feature engineering ──
    X, stage_enc, disease_enc, feature_cols = engineer_features(df)
    y = df["yield_kg_acre"].values

    # ── 5-fold CV ──
    log.info(f"Running {TRAIN_CFG['cv_folds']}-fold cross-validation...")
    model_cv = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
    kf = KFold(n_splits=TRAIN_CFG["cv_folds"], shuffle=True, random_state=42)

    cv_mae = -cross_val_score(model_cv, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(model_cv, X, y, cv=kf, scoring="r2")

    log.info(f"CV MAE:  {cv_mae.mean():.2f} ± {cv_mae.std():.2f} kg/acre")
    log.info(f"CV R²:   {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

    # ── Final model on full data ──
    log.info("Fitting final model on full dataset...")
    final_model = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
    final_model.fit(X, y)

    # Full-data metrics (for sanity check, not evaluation)
    y_pred = final_model.predict(X)
    train_mae = mean_absolute_error(y, y_pred)
    train_r2 = r2_score(y, y_pred)
    log.info(f"Train MAE: {train_mae:.2f} kg/acre  |  Train R²: {train_r2:.4f}")

    # ── Feature importances ──
    importances = dict(zip(feature_cols, final_model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    log.info("Feature importances:")
    for feat, imp in sorted_imp:
        bar = "█" * int(imp * 40)
        log.info(f"  {feat:<28} {imp:.4f}  {bar}")

    # ── Save model bundle ──
    # Bundle everything the inference service needs
    bundle = {
        "model":        final_model,
        "stage_enc":    stage_enc,
        "disease_enc":  disease_enc,
        "feature_cols": feature_cols,
        "crop_name":    crop_name,
        "cv_mae_mean":  float(cv_mae.mean()),
        "cv_mae_std":   float(cv_mae.std()),
        "cv_r2_mean":   float(cv_r2.mean()),
        "crop_config":  CFG["crops"][crop_name],
    }

    out_path = MODELS_DIR / f"{crop_name}_yield_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    log.info(f"Model saved → {out_path}")

    # ── Optional plots ──
    if show_plots:
        _plot_results(crop_name, y, y_pred, sorted_imp)

    return {
        "crop": crop_name,
        "cv_mae": round(float(cv_mae.mean()), 2),
        "cv_r2": round(float(cv_r2.mean()), 4),
        "samples": n,
        "saved_to": str(out_path),
    }


def _plot_results(crop_name, y_true, y_pred, feature_importances):
    """Generate training diagnostic plots (optional dependency)."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"{crop_name.capitalize()} Yield Model — Training Diagnostics", fontsize=13)

        # Predicted vs Actual
        ax = axes[0]
        ax.scatter(y_true, y_pred, alpha=0.3, s=8, color="#2196F3")
        mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual yield (kg/acre)")
        ax.set_ylabel("Predicted yield (kg/acre)")
        ax.set_title("Predicted vs Actual")
        ax.legend()

        # Feature importance bar chart
        ax = axes[1]
        feats, imps = zip(*feature_importances[:8])  # top 8
        colors = ["#4CAF50" if i == 0 else "#2196F3" for i in range(len(feats))]
        ax.barh(list(feats)[::-1], list(imps)[::-1], color=colors[::-1])
        ax.set_xlabel("Importance score")
        ax.set_title("Top Feature Importances")

        plt.tight_layout()
        plot_path = MODELS_DIR / f"{crop_name}_yield_model_diagnostics.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        log.info(f"Diagnostic plot saved → {plot_path}")
        plt.close()

    except ImportError:
        log.warning("matplotlib not available — skipping plots. Install with: pip install matplotlib")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Agri-Vision yield models")
    parser.add_argument(
        "--crop",
        choices=["cotton", "tomato", "potato", "all"],
        default="all",
        help="Which crop model to train (default: all)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib diagnostic plots",
    )
    args = parser.parse_args()

    crops = ["cotton", "tomato", "potato"] if args.crop == "all" else [args.crop]
    show_plots = not args.no_plots

    results = []
    for crop in crops:
        try:
            meta = train_crop_model(crop, show_plots=show_plots)
            results.append(meta)
        except Exception as e:
            log.error(f"Failed to train {crop} model: {e}", exc_info=True)

    # Summary table
    log.info("\n" + "="*60)
    log.info("TRAINING SUMMARY")
    log.info("="*60)
    log.info(f"{'Crop':<10} {'CV MAE (kg/acre)':<20} {'CV R²':<10} {'Samples'}")
    log.info("-"*60)
    for r in results:
        log.info(f"{r['crop']:<10} {r['cv_mae']:<20.2f} {r['cv_r2']:<10.4f} {r['samples']}")
    log.info("="*60)
    log.info("All models saved to ai_models/")


if __name__ == "__main__":
    main()