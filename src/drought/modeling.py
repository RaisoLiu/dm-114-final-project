from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from .features import DATE_ORDINAL_COL, REGION_COL, date_ordinals


class FeaturePreprocessor:
    def __init__(self) -> None:
        self.region_levels: list[str] = []
        self.medians: pd.Series | None = None
        self.columns: list[str] = []
        self.date_origin: int | None = None

    def fit(self, features: pd.DataFrame, meta: pd.DataFrame) -> "FeaturePreprocessor":
        self.region_levels = sorted(meta[REGION_COL].astype(str).unique().tolist())
        self.date_origin = int(self._meta_date_ordinals(meta).min())
        matrix = self._build_matrix(features, meta)
        self.medians = matrix.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        self.columns = matrix.columns.tolist()
        return self

    def transform(self, features: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        if self.medians is None or self.date_origin is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform.")
        matrix = self._build_matrix(features, meta)
        for col in self.columns:
            if col not in matrix.columns:
                matrix[col] = np.nan
        matrix = matrix[self.columns]
        matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        return matrix.astype(float)

    def _meta_date_ordinals(self, meta: pd.DataFrame) -> pd.Series:
        if DATE_ORDINAL_COL in meta.columns:
            return pd.to_numeric(meta[DATE_ORDINAL_COL], errors="raise").astype("int64").reset_index(drop=True)
        return date_ordinals(meta["end_date"]).reset_index(drop=True)

    def _build_matrix(self, features: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
        matrix = features.drop(columns=["region_id_raw"], errors="ignore").copy()
        matrix = matrix.apply(pd.to_numeric, errors="coerce")

        region = meta[REGION_COL].astype(str).reset_index(drop=True)
        level_to_code = {level: i for i, level in enumerate(self.region_levels)}
        matrix["region_code"] = region.map(level_to_code).fillna(-1).astype(float).to_numpy()
        for i, level in enumerate(self.region_levels):
            matrix[f"region_ohe_{i:03d}"] = (region == level).astype(float).to_numpy()

        if self.date_origin is not None:
            ordinals = self._meta_date_ordinals(meta)
            matrix["end_days_since_origin"] = (ordinals - self.date_origin).astype(float).to_numpy()

        return matrix


@dataclass
class HorizonEnsemble:
    models_by_horizon: list[list[tuple[str, Any]]]
    weights_by_horizon: list[np.ndarray]
    model_names: list[str]

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        horizon_predictions: list[np.ndarray] = []
        for models, weights in zip(self.models_by_horizon, self.weights_by_horizon, strict=True):
            preds = np.column_stack([model.predict(x) for _, model in models])
            horizon_predictions.append(preds @ weights)
        return np.column_stack(horizon_predictions)


def available_model_names(requested: str) -> list[str]:
    if requested == "auto":
        names = ["hgb", "extra"]
        try:
            import lightgbm  # noqa: F401

            names.insert(0, "lightgbm")
        except Exception:
            pass
        return names
    return [name.strip() for name in requested.split(",") if name.strip()]


def make_model(name: str, seed: int, fast: bool) -> Any:
    if name == "hgb":
        return HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.045 if not fast else 0.08,
            max_iter=900 if not fast else 160,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.03,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=40 if not fast else 12,
            random_state=seed,
        )
    if name == "extra":
        return ExtraTreesRegressor(
            n_estimators=650 if not fast else 120,
            min_samples_leaf=2,
            max_features=0.70,
            bootstrap=False,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=450 if not fast else 80,
            min_samples_leaf=3,
            max_features=0.75,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as exc:
            raise RuntimeError("lightgbm was requested but is not installed.") from exc
        return LGBMRegressor(
            objective="mae",
            n_estimators=2400 if not fast else 350,
            learning_rate=0.025 if not fast else 0.06,
            num_leaves=63,
            min_child_samples=25,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.80,
            reg_alpha=0.05,
            reg_lambda=0.30,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unknown model name: {name}")


def model_weight_from_mae(maes: list[float]) -> np.ndarray:
    values = np.asarray(maes, dtype=float)
    values = np.where(np.isfinite(values), values, np.nanmax(values[np.isfinite(values)]) if np.isfinite(values).any() else 1.0)
    inv = 1.0 / np.maximum(values, 1e-6)
    return inv / inv.sum()


def fit_horizon_ensemble(
    x: pd.DataFrame,
    y: pd.DataFrame,
    model_names: list[str],
    seed: int,
    fast: bool,
    sample_weight: np.ndarray | None = None,
    validation: tuple[pd.DataFrame, pd.DataFrame] | None = None,
    fixed_weights: list[np.ndarray] | None = None,
) -> tuple[HorizonEnsemble, dict[str, Any]]:
    models_by_horizon: list[list[tuple[str, Any]]] = []
    weights_by_horizon: list[np.ndarray] = []
    metrics: dict[str, Any] = {"per_horizon": []}

    for horizon_idx, target_col in enumerate(y.columns):
        horizon_models: list[tuple[str, Any]] = []
        horizon_maes: list[float] = []
        for model_idx, name in enumerate(model_names):
            model = make_model(name, seed + 101 * horizon_idx + 17 * model_idx, fast)
            fit_kwargs: dict[str, Any] = {}
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            model.fit(x, y[target_col].to_numpy(dtype=float), **fit_kwargs)
            horizon_models.append((name, model))

            if validation is not None:
                x_val, y_val = validation
                pred = np.clip(model.predict(x_val), 0.0, 5.0)
                horizon_maes.append(float(mean_absolute_error(y_val[target_col], pred)))

        if fixed_weights is not None:
            weights = fixed_weights[horizon_idx]
        elif validation is not None:
            weights = model_weight_from_mae(horizon_maes)
        else:
            weights = np.full(len(horizon_models), 1.0 / len(horizon_models))

        models_by_horizon.append(horizon_models)
        weights_by_horizon.append(weights)
        metrics["per_horizon"].append(
            {
                "target": target_col,
                "model_mae": dict(zip(model_names, horizon_maes, strict=False)) if horizon_maes else {},
                "weights": dict(zip(model_names, weights.tolist(), strict=False)),
            }
        )

    ensemble = HorizonEnsemble(models_by_horizon, weights_by_horizon, model_names)
    if validation is not None:
        x_val, y_val = validation
        pred = np.clip(ensemble.predict(x_val), 0.0, 5.0)
        metrics["mae"] = float(mean_absolute_error(y_val.to_numpy().ravel(), pred.ravel()))
        metrics["horizon_mae"] = {
            col: float(mean_absolute_error(y_val[col], pred[:, i])) for i, col in enumerate(y_val.columns)
        }
    return ensemble, metrics


def recency_weights(meta: pd.DataFrame, half_life_days: float) -> np.ndarray | None:
    if half_life_days <= 0:
        return None
    dates = pd.to_datetime(meta["end_date"])
    age_days = (dates.max() - dates).dt.days.to_numpy(dtype=float)
    weights = np.power(0.5, age_days / half_life_days)
    return weights / np.mean(weights)


def save_bundle(path: str, bundle: dict[str, Any]) -> None:
    joblib.dump(bundle, path)
