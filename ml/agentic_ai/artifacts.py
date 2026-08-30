from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from .config import MODEL_LAYOUT


class ArtifactError(RuntimeError):
    pass


class ToolInputError(ValueError):
    pass


@dataclass(frozen=True)
class FeatureDef:
    name: str
    dtype: str | None = None
    description: str | None = None

    def json_type(self) -> str:
        d = (self.dtype or "").lower()
        if any(x in d for x in ("bool", "boolean")):
            return "boolean"
        if any(x in d for x in ("int", "integer")):
            return "integer"
        if any(x in d for x in ("float", "double", "number", "numeric", "decimal")):
            return "number"
        return "string" if any(x in d for x in ("str", "object", "category", "string")) else "number"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    task: str
    folder: Path
    champion: Path
    schema: Path
    metadata: Path | None
    manifest: Path | None
    metrics: Path | None
    policy: Path | None
    reference: Path | None
    importance: Path | None
    requirements: Path | None
    profiles: Path | None = None
    profile_csv: Path | None = None
    summary_csv: Path | None = None



CHAMPION_CANDIDATES = {
    "pd": [
        "pd_champion_new.joblib",
        "pd_champion.joblib",
    ],
    "ews": [
        "ews_xgboost_champion.joblib",
        "ews_logistic_champion.joblib",
    ],
    "lgd": [
        "final_lgd_xgboost_new.pkl",
        "lgd_champion.joblib",
        "final_lgd_xgboost.pkl",
    ],
    "pd_cluster": [
        "pd_cluster_champion.joblib",
    ],
}


def _resolve_champion(
    key: str,
    folder: Path,
    configured: str,
) -> Path:
    candidates = [configured] + CHAMPION_CANDIDATES.get(key, [])
    seen: set[str] = set()

    for name in candidates:
        if not name or name in seen:
            continue
        seen.add(name)
        path = folder / name
        if path.exists():
            return path

    return folder / configured

def _path(folder: Path, value: str | None) -> Path | None:
    return None if not value else folder / value


def spec_for(key: str) -> ModelSpec:
    if key not in MODEL_LAYOUT:
        raise ArtifactError(f"Model key {key!r} tidak dikenal.")
    cfg = MODEL_LAYOUT[key]
    folder = Path(cfg["folder"])
    return ModelSpec(
        key=key,
        task=cfg["task"],
        folder=folder,
        champion=_resolve_champion(
            key,
            folder,
            cfg["champion"],
        ),
        schema=folder / cfg["schema"],
        metadata=_path(folder, cfg.get("metadata")),
        manifest=_path(folder, cfg.get("manifest")),
        metrics=_path(folder, cfg.get("metrics")),
        policy=_path(folder, cfg.get("policy")),
        reference=_path(folder, cfg.get("reference")),
        importance=_path(folder, cfg.get("importance")),
        requirements=_path(folder, cfg.get("requirements")),
        profiles=_path(folder, cfg.get("profiles")),
        profile_csv=_path(folder, cfg.get("profile_csv")),
        summary_csv=_path(folder, cfg.get("summary_csv")),
    )


def _read_json(path: Path | None) -> Any:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactError(f"Gagal membaca JSON {path}: {exc}") from exc


def _first_present(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in mapping and mapping[k] is not None:
            return mapping[k]
    return None


def _recursive_find(data: Any, keys: set[str]) -> Any:
    if isinstance(data, dict):
        for k, v in data.items():
            if str(k).lower() in keys and v is not None:
                return v
        for v in data.values():
            found = _recursive_find(v, keys)
            if found is not None:
                return found
    elif isinstance(data, list):
        for v in data:
            found = _recursive_find(v, keys)
            if found is not None:
                return found
    return None


def _coerce_feature_defs(value: Any) -> list[FeatureDef]:
    out: list[FeatureDef] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                out.append(FeatureDef(item))
            elif isinstance(item, dict):
                name = _first_present(item, ("name", "feature", "column", "feature_name"))
                if name:
                    out.append(
                        FeatureDef(
                            str(name),
                            str(_first_present(item, ("dtype", "type", "data_type")) or ""),
                            str(_first_present(item, ("description", "desc")) or "") or None,
                        )
                    )
    elif isinstance(value, dict):
        # Typical form: {"feature_a": "float64", "feature_b": "int64"}
        ignored = {
            "target", "label", "version", "schema_version", "required", "optional",
            "model", "model_name", "model_version", "algorithm", "task",
            "n_clusters", "created_at", "trained_at",
        }
        for name, meta in value.items():
            if str(name).lower() in ignored:
                continue
            if isinstance(meta, str):
                out.append(FeatureDef(str(name), meta))
            elif isinstance(meta, dict):
                dtype = _first_present(meta, ("dtype", "type", "data_type"))
                desc = _first_present(meta, ("description", "desc"))
                # Only accept dict entries that look like feature metadata.
                if dtype is not None or desc is not None:
                    out.append(FeatureDef(str(name), str(dtype or ""), str(desc or "") or None))
    return out


def _dedupe_features(features: list[FeatureDef]) -> list[FeatureDef]:
    # Preserve first-seen order, but enrich an earlier name-only entry with dtype/
    # description found later (common when schema has both feature_names + dtypes).
    order: list[str] = []
    merged: dict[str, FeatureDef] = {}
    for f in features:
        if not f.name:
            continue
        if f.name not in merged:
            order.append(f.name)
            merged[f.name] = f
        else:
            old = merged[f.name]
            merged[f.name] = FeatureDef(
                name=f.name,
                dtype=old.dtype or f.dtype,
                description=old.description or f.description,
            )
    return [merged[name] for name in order]


def _looks_like_metadata_only(features: list[FeatureDef]) -> bool:
    """Reject sidecar JSON that is actually model metadata, not a feature schema.

    This matters for PD Cluster where an early sidecar could contain only
    ``model`` and ``model_version``. Those are artifact metadata and must never
    be sent to KMeans as borrower features.
    """
    if not features:
        return True
    metadata_names = {
        "model", "model_name", "model_version", "version", "algorithm",
        "task", "n_clusters", "schema_version", "created_at", "trained_at",
    }
    names = {f.name.strip().lower() for f in features}
    return bool(names) and names.issubset(metadata_names)


class ArtifactStore:
    """Lazy loader for all saved model artifacts and sidecar metadata."""

    @lru_cache(maxsize=8)
    def bundle(self, key: str) -> Any:
        spec = spec_for(key)
        if not spec.champion.exists():
            raise ArtifactError(f"Champion artifact tidak ditemukan: {spec.champion}")
        try:
            import joblib

            return joblib.load(spec.champion)
        except Exception as exc:
            raise ArtifactError(
                f"Gagal load {spec.champion}. Pastikan versi library sesuai *_requirements.txt. "
                f"Detail: {exc}"
            ) from exc

    @lru_cache(maxsize=16)
    def metadata(self, key: str) -> dict[str, Any]:
        data = _read_json(spec_for(key).metadata)
        return data if isinstance(data, dict) else {}

    @lru_cache(maxsize=16)
    def manifest(self, key: str) -> dict[str, Any]:
        data = _read_json(spec_for(key).manifest)
        return data if isinstance(data, dict) else {}

    @lru_cache(maxsize=16)
    def policy(self, key: str) -> dict[str, Any]:
        data = _read_json(spec_for(key).policy)
        return data if isinstance(data, dict) else {}

    @lru_cache(maxsize=16)
    def reference(self, key: str) -> dict[str, Any]:
        data = _read_json(spec_for(key).reference)
        return data if isinstance(data, dict) else {}

    @lru_cache(maxsize=16)
    def schema_raw(self, key: str) -> Any:
        spec = spec_for(key)
        if not spec.schema.exists():
            raise ArtifactError(f"Feature schema tidak ditemukan: {spec.schema}")
        return _read_json(spec.schema)

    @lru_cache(maxsize=16)
    def feature_defs(self, key: str) -> tuple[FeatureDef, ...]:
        """Return the actual borrower feature contract for a saved artifact.

        For clustering we prefer the feature list stored in the joblib bundle,
        because some sidecar JSON files are model metadata rather than schemas.
        For supervised models, sidecar schema remains the first choice.
        """

        def from_bundle() -> list[FeatureDef]:
            try:
                bundle = self.bundle(key)
            except Exception:
                return []
            if not isinstance(bundle, dict):
                names = getattr(bundle, "feature_names_in_", None)
                return _coerce_feature_defs(list(names)) if names is not None else []
            raw_features = _first_present(
                bundle,
                (
                    "features", "risk_features", "feature_names", "columns",
                    "feature_list", "input_features",
                ),
            )
            return _dedupe_features(_coerce_feature_defs(raw_features))

        # PD Cluster's feature contract is most reliably carried in the bundle.
        if key == "pd_cluster":
            bundle_features = from_bundle()
            if bundle_features and not _looks_like_metadata_only(bundle_features):
                return tuple(bundle_features)

        raw = self.schema_raw(key)
        candidates: list[FeatureDef] = []
        if isinstance(raw, dict):
            for k in (
                "features", "feature_names", "columns", "inputs",
                "feature_schema", "dtypes", "risk_features",
            ):
                if k in raw:
                    candidates.extend(_coerce_feature_defs(raw[k]))
            if not candidates:
                candidates.extend(_coerce_feature_defs(raw))
        else:
            candidates.extend(_coerce_feature_defs(raw))

        result = _dedupe_features(candidates)
        if result and not _looks_like_metadata_only(result):
            return tuple(result)

        # Fallback for bundles that stored feature names only inside joblib.
        bundle_features = from_bundle()
        if bundle_features and not _looks_like_metadata_only(bundle_features):
            return tuple(bundle_features)

        raise ArtifactError(
            f"Tidak bisa mengekstrak feature names aktual untuk {key}. "
            f"Sidecar {spec_for(key).schema} tampak bukan feature schema. "
            "Pastikan joblib bundle menyimpan key `features` / `risk_features`."
        )

    def feature_names(self, key: str) -> list[str]:
        return [f.name for f in self.feature_defs(key)]

    @staticmethod
    def _is_missing_value(value: Any) -> bool:
        if value is None:
            return True
        try:
            import pandas as pd
            missing = pd.isna(value)
            return bool(missing) if not hasattr(missing, "__len__") else False
        except Exception:
            return False

    def input_quality(self, key: str, features: dict[str, Any]) -> dict[str, Any]:
        expected = self.feature_names(key)
        missing = [
            name for name in expected
            if name not in features or self._is_missing_value(features.get(name))
        ]
        observed = len(expected) - len(missing)
        ratio = observed / len(expected) if expected else 0.0
        if ratio >= 0.75:
            band = "high"
        elif ratio >= 0.50:
            band = "medium"
        else:
            band = "low"
        return {
            "expected_feature_count": len(expected),
            "observed_feature_count": observed,
            "missing_feature_count": len(missing),
            "feature_completeness": round(ratio, 4),
            "feature_completeness_percent": round(ratio * 100, 2),
            "data_completeness_band": band,
            "missing_features_imputed_by_saved_pipeline": missing,
            "warning": (
                None if not missing else
                "Prediction tetap dijalankan; feature yang tidak tersedia dikirim sebagai NaN "
                "dan ditangani preprocessing/imputer yang tersimpan bersama model. "
                "Semakin banyak feature yang diimputasi, semakin hati-hati interpretasinya."
            ),
        }

    def _frame(self, key: str, features: dict[str, Any]):
        """Create a complete model frame and pad unavailable inputs with NaN.

        The saved preprocessing pipeline must own imputation. The agent must not
        invent zeros/medians/modes itself.
        """
        import numpy as np
        import pandas as pd

        defs = list(self.feature_defs(key))
        names = [f.name for f in defs]
        row = {
            name: (
                np.nan
                if name not in features or self._is_missing_value(features.get(name))
                else features[name]
            )
            for name in names
        }
        df = pd.DataFrame([row], columns=names)

        # Preserve broad training-time type families. In particular, a missing
        # categorical column should remain object-like so a fitted
        # most_frequent/OHE pipeline can impute the learned category instead of
        # receiving a float-only placeholder column.
        for f in defs:
            d = (f.dtype or "").lower()
            if any(x in d for x in ("object", "str", "string", "category")):
                df[f.name] = df[f.name].astype("object")
            elif any(x in d for x in ("bool", "boolean")):
                # Leave NaN possible; pandas nullable boolean keeps missing state.
                try:
                    df[f.name] = df[f.name].astype("boolean")
                except Exception:
                    pass
            elif any(x in d for x in ("int", "float", "double", "numeric", "number", "decimal")):
                df[f.name] = pd.to_numeric(df[f.name], errors="coerce")
        return df

    @staticmethod
    def _estimator(bundle: Any) -> Any:
        if not isinstance(bundle, dict):
            return bundle
        est = _first_present(
            bundle,
            (
                "model",
                "estimator",
                "pipeline",
                "champion",
                "classifier",
                "regressor",
                "clusterer",
                "kmeans",
            ),
        )
        if est is None:
            raise ArtifactError(f"Bundle dict tidak punya estimator. Keys: {sorted(bundle.keys())}")
        return est

    @staticmethod
    def _positive_class(bundle: Any, metadata: dict[str, Any]) -> Any:
        if isinstance(bundle, dict):
            val = _first_present(bundle, ("positive_class", "positive_label"))
            if val is not None:
                return val
        val = _recursive_find(metadata, {"positive_class", "positive_label"})
        return 1 if val is None else val

    @staticmethod
    def _positive_meaning(bundle: Any, metadata: dict[str, Any]) -> str | None:
        if isinstance(bundle, dict):
            val = _first_present(bundle, ("positive_class_meaning", "positive_label_meaning"))
            if val is not None:
                return str(val)
        val = _recursive_find(metadata, {"positive_class_meaning", "positive_label_meaning"})
        return None if val is None else str(val)

    def _threshold(self, key: str, bundle: Any) -> float | None:
        candidates = []
        if isinstance(bundle, dict):
            candidates.append(
                _first_present(bundle, ("threshold", "decision_threshold", "classification_threshold"))
            )
        candidates.append(
            _recursive_find(
                self.policy(key),
                {"threshold", "decision_threshold", "classification_threshold", "operating_threshold"},
            )
        )
        candidates.append(
            _recursive_find(
                self.metadata(key),
                {"threshold", "decision_threshold", "classification_threshold", "operating_threshold"},
            )
        )
        for val in candidates:
            try:
                if val is not None:
                    return float(val)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _probability(estimator: Any, X, positive_class: Any) -> float:
        if not hasattr(estimator, "predict_proba"):
            raise ArtifactError("Classifier tidak memiliki predict_proba().")
        proba = estimator.predict_proba(X)
        if getattr(proba, "ndim", 1) != 2:
            raise ArtifactError(f"Bentuk predict_proba tidak dikenali: {getattr(proba, 'shape', None)}")
        classes = getattr(estimator, "classes_", None)
        idx = 1 if proba.shape[1] > 1 else 0
        if classes is not None:
            try:
                classes_list = list(classes)
                if positive_class in classes_list:
                    idx = classes_list.index(positive_class)
                elif str(positive_class) in [str(x) for x in classes_list]:
                    idx = [str(x) for x in classes_list].index(str(positive_class))
            except Exception:
                pass
        return float(proba[0, idx])

    @staticmethod
    def _apply_calibrator(key: str, bundle: Any, raw_probability: float) -> tuple[float, str, str | None]:
        if not isinstance(bundle, dict) or bundle.get("calibrator") is None:
            return raw_probability, "raw_model_probability", None
        calibrator = bundle["calibrator"]
        try:
            import numpy as np

            # The saved PD notebook fits Platt calibration on logit(raw PD), not
            # directly on raw probability. Preserve that exact training contract.
            if key == "pd":
                clipped = float(np.clip(raw_probability, 1e-6, 1 - 1e-6))
                value = math.log(clipped / (1.0 - clipped))
                arr = np.array([[value]], dtype=float)
                source = "separate_calibrator_on_logit_raw_probability"
            else:
                arr = np.array([[raw_probability]], dtype=float)
                source = "separate_calibrator"

            if hasattr(calibrator, "predict_proba"):
                out = calibrator.predict_proba(arr)
                idx = 1 if out.shape[1] > 1 else 0
                return float(out[0, idx]), source, None
            if hasattr(calibrator, "predict"):
                out = calibrator.predict(arr)
                return float(out[0]), source, None
        except Exception as exc:
            return raw_probability, "raw_model_probability", f"Calibrator gagal diterapkan: {exc}"
        return raw_probability, "raw_model_probability", "Objek calibrator tidak punya predict/predict_proba."

    @staticmethod
    def _metadata_excerpt(metadata: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "model_name",
            "champion_name",
            "target",
            "target_name",
            "horizon",
            "model_version",
            "semantic_rules_version",
            "training_period",
        )
        out: dict[str, Any] = {}
        for source in (metadata, manifest):
            for k in keys:
                if k in source and k not in out:
                    out[k] = source[k]
        return out

    @staticmethod
    def _to_dense_row(Xt):
        try:
            import scipy.sparse as sp

            if sp.issparse(Xt):
                return Xt.toarray()
        except Exception:
            pass
        try:
            return Xt.to_numpy()
        except Exception:
            return Xt

    def _local_explanation(self, key: str, estimator: Any, X, top_n: int = 5) -> dict[str, Any]:
        """Best-effort local contribution explanation.

        For XGBoost this uses pred_contribs in model-margin space. For a linear
        classifier/regressor it uses coefficient * transformed feature value.
        These are model contributions, not causal explanations.
        """
        try:
            final = estimator
            Xt = X
            names = list(X.columns)

            if hasattr(estimator, "steps") and len(estimator.steps) >= 1:
                final = estimator.steps[-1][1]
                if len(estimator.steps) > 1:
                    prep = estimator[:-1]
                    Xt = prep.transform(X)
                    try:
                        names = [str(x) for x in prep.get_feature_names_out()]
                    except Exception:
                        dense = self._to_dense_row(Xt)
                        width = int(getattr(dense, "shape", [1, len(names)])[1])
                        names = [f"x{i}" for i in range(width)]

            dense = self._to_dense_row(Xt)

            if hasattr(final, "get_booster"):
                import xgboost as xgb

                booster = final.get_booster()
                contrib = booster.predict(xgb.DMatrix(Xt), pred_contribs=True)[0]
                # Last value is bias/base contribution.
                values = contrib[:-1]
                if len(names) != len(values):
                    names = [f"x{i}" for i in range(len(values))]
                ranked = sorted(
                    zip(names, values), key=lambda z: abs(float(z[1])), reverse=True
                )[:top_n]
                return {
                    "available": True,
                    "scope": "local_customer",
                    "basis": "xgboost_pred_contribs_model_margin",
                    "top_factors": [
                        {
                            "feature": n,
                            "contribution": round(float(v), 6),
                            "direction": "increase_model_score" if float(v) > 0 else "decrease_model_score",
                        }
                        for n, v in ranked
                    ],
                    "warning": "Kontribusi berada pada ruang skor/model margin dan bukan bukti kausal.",
                }

            if hasattr(final, "coef_"):
                import numpy as np

                arr = np.asarray(dense)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                coef = np.asarray(final.coef_)
                coef_row = coef[0] if coef.ndim > 1 else coef
                if arr.shape[1] != coef_row.shape[0]:
                    raise ValueError("Dimensi transformed features dan koefisien tidak sama.")
                values = arr[0] * coef_row
                if len(names) != len(values):
                    names = [f"x{i}" for i in range(len(values))]
                ranked = sorted(
                    zip(names, values), key=lambda z: abs(float(z[1])), reverse=True
                )[:top_n]
                return {
                    "available": True,
                    "scope": "local_customer",
                    "basis": "linear_coefficient_times_transformed_value",
                    "top_factors": [
                        {
                            "feature": n,
                            "contribution": round(float(v), 6),
                            "direction": "increase_model_score" if float(v) > 0 else "decrease_model_score",
                        }
                        for n, v in ranked
                    ],
                    "warning": "Kontribusi adalah kontribusi matematis model, bukan bukti kausal.",
                }
        except Exception as exc:
            local_error = str(exc)
        else:
            local_error = "Estimator tidak mendukung local contribution adapter."

        # Fallback: global feature importance. This must never be narrated as an
        # individual reason for this specific customer.
        spec = spec_for(key)
        if spec.importance is not None and spec.importance.exists():
            try:
                import pandas as pd

                df = pd.read_csv(spec.importance)
                if len(df.columns) >= 2:
                    feature_col = next(
                        (c for c in df.columns if "feature" in c.lower()), df.columns[0]
                    )
                    importance_col = next(
                        (c for c in df.columns if "importance" in c.lower()), df.columns[1]
                    )
                    top = df.sort_values(importance_col, ascending=False).head(top_n)
                    return {
                        "available": True,
                        "scope": "global_model",
                        "basis": "saved_feature_importance_csv",
                        "top_factors": [
                            {
                                "feature": str(r[feature_col]),
                                "importance": float(r[importance_col]),
                            }
                            for _, r in top.iterrows()
                        ],
                        "warning": (
                            "Ini feature importance global, bukan alasan individual customer. "
                            f"Local explanation tidak tersedia: {local_error}"
                        ),
                    }
            except Exception as exc:
                local_error = f"{local_error}; fallback importance gagal: {exc}"

        return {
            "available": False,
            "scope": "none",
            "basis": None,
            "top_factors": [],
            "warning": local_error,
        }

    def _input_reference_warnings(self, key: str, features: dict[str, Any]) -> list[str]:
        ref = self.reference(key)
        if not isinstance(ref, dict) or not ref:
            return []
        feature_stats = ref.get("features", ref)
        if not isinstance(feature_stats, dict):
            return []
        warnings: list[str] = []
        for name, value in features.items():
            if value is None or name not in feature_stats or not isinstance(feature_stats[name], dict):
                continue
            try:
                x = float(value)
            except (TypeError, ValueError):
                continue
            stat = feature_stats[name]
            lo = _first_present(stat, ("min", "minimum", "p01"))
            hi = _first_present(stat, ("max", "maximum", "p99"))
            try:
                if lo is not None and x < float(lo):
                    warnings.append(f"{name}={x} di bawah reference minimum {lo}")
                if hi is not None and x > float(hi):
                    warnings.append(f"{name}={x} di atas reference maximum {hi}")
            except (TypeError, ValueError):
                pass
        return warnings[:10]

    def predict_classifier(self, key: str, features: dict[str, Any]) -> dict[str, Any]:
        bundle = self.bundle(key)
        estimator = self._estimator(bundle)
        X = self._frame(key, features)
        metadata = self.metadata(key)
        positive_class = self._positive_class(bundle, metadata)
        raw_p = self._probability(estimator, X, positive_class)
        p, source, calibrator_warning = self._apply_calibrator(key, bundle, raw_p)
        threshold = self._threshold(key, bundle)

        output: dict[str, Any] = {
            "model": key,
            "task": "classification",
            "probability": round(p, 8),
            "probability_percent": round(p * 100, 4),
            "raw_model_probability": round(raw_p, 8),
            "probability_source": source,
            "threshold": None if threshold is None else round(threshold, 8),
            "predicted_positive": None if threshold is None else bool(p >= threshold),
            "positive_class": positive_class,
            "positive_class_meaning": self._positive_meaning(bundle, metadata),
            "artifact_info": self._metadata_excerpt(metadata, self.manifest(key)),
            "input_quality": self.input_quality(key, features),
            "input_reference_warnings": self._input_reference_warnings(key, features),
            "explanation": self._local_explanation(key, estimator, X),
        }
        if calibrator_warning:
            output["calibrator_warning"] = calibrator_warning
        if isinstance(bundle, dict) and bundle.get("risk_cutoffs") is not None:
            # Preserve as provenance only. Do not infer band semantics here.
            output["saved_risk_cutoffs"] = bundle.get("risk_cutoffs")
        return output

    def predict_regression(self, key: str, features: dict[str, Any]) -> dict[str, Any]:
        bundle = self.bundle(key)
        estimator = self._estimator(bundle)
        X = self._frame(key, features)
        if not hasattr(estimator, "predict"):
            raise ArtifactError("Regressor tidak memiliki predict().")
        pred = float(estimator.predict(X)[0])
        output: dict[str, Any] = {
            "model": key,
            "task": "regression",
            "prediction": round(pred, 8),
            "artifact_info": self._metadata_excerpt(self.metadata(key), self.manifest(key)),
            "input_quality": self.input_quality(key, features),
            "input_reference_warnings": self._input_reference_warnings(key, features),
            "explanation": self._local_explanation(key, estimator, X),
        }
        # LGD is normally a rate. Only add a human-readable percentage when the
        # saved model output is already on [0, 1]; do not silently rescale 0-100.
        if 0 <= pred <= 1:
            output["prediction_percent"] = round(pred * 100, 4)
        else:
            output["scale_warning"] = (
                "Prediction berada di luar [0,1]. Cek metadata target/transform; "
                "pipeline tidak melakukan rescaling otomatis."
            )
        return output

    def _cluster_profile(self, key: str, cluster: Any) -> Any:
        spec = spec_for(key)
        if spec.profiles and spec.profiles.exists():
            data = _read_json(spec.profiles)
            if isinstance(data, dict):
                for candidate in (cluster, str(cluster), f"cluster_{cluster}"):
                    if candidate in data:
                        return data[candidate]
                for k, v in data.items():
                    if str(k).lower().replace("cluster_", "") == str(cluster):
                        return v
            if isinstance(data, list):
                for row in data:
                    if not isinstance(row, dict):
                        continue
                    value = _first_present(row, ("cluster", "cluster_id", "label"))
                    if value is not None and str(value) == str(cluster):
                        return row
        for csv_path in (spec.profile_csv, spec.summary_csv):
            if csv_path and csv_path.exists():
                try:
                    import pandas as pd

                    df = pd.read_csv(csv_path)
                    cluster_col = next(
                        (c for c in df.columns if "cluster" in c.lower()), None
                    )
                    if cluster_col:
                        hit = df[df[cluster_col].astype(str) == str(cluster)]
                        if not hit.empty:
                            return hit.iloc[0].to_dict()
                except Exception:
                    pass
        return None

    def predict_cluster(self, key: str, features: dict[str, Any]) -> dict[str, Any]:
        bundle = self.bundle(key)
        X = self._frame(key, features)

        if not isinstance(bundle, dict):
            model = bundle
            if not hasattr(model, "predict"):
                raise ArtifactError("Cluster champion tidak memiliki predict().")
            cluster = model.predict(X)[0]
        else:
            # Support common saved patterns: full pipeline, or scaler -> PCA -> KMeans.
            if bundle.get("pipeline") is not None and hasattr(bundle["pipeline"], "predict"):
                cluster = bundle["pipeline"].predict(X)[0]
            else:
                model = _first_present(bundle, ("kmeans", "clusterer", "model", "estimator"))
                if model is None or not hasattr(model, "predict"):
                    raise ArtifactError(
                        f"Bundle cluster tidak punya predictor. Keys: {sorted(bundle.keys())}"
                    )
                import numpy as np
                import pandas as pd

                Xt = X.copy()

                # Match the saved clustering notebook: ordinal rating -> numeric
                # coercion -> training winsor bounds -> median imputer -> scaler
                # -> PCA -> KMeans. Each object should ideally be saved in bundle.
                rating_map = bundle.get("rating_map") or {
                    "AAA": 1, "AA": 2, "A": 3, "BBB": 4, "BB": 5,
                    "B": 6, "CCC": 7, "CC": 8, "C": 9, "D": 10,
                }
                if "app_rating_internal" in Xt.columns:
                    Xt["app_rating_internal"] = Xt["app_rating_internal"].map(rating_map)

                for col in Xt.columns:
                    Xt[col] = (
                        pd.to_numeric(Xt[col], errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .astype(float)
                    )

                clip_bounds = _first_present(
                    bundle, ("clip_bounds", "winsor_bounds", "winsorization_bounds")
                )
                if isinstance(clip_bounds, dict):
                    for col, bounds in clip_bounds.items():
                        if col not in Xt.columns or not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                            continue
                        Xt[col] = Xt[col].clip(float(bounds[0]), float(bounds[1]))

                imputer = bundle.get("imputer")
                if imputer is not None and hasattr(imputer, "transform"):
                    Xt = imputer.transform(Xt)

                scaler = bundle.get("scaler")
                if scaler is not None and hasattr(scaler, "transform"):
                    Xt = scaler.transform(Xt)
                else:
                    transformer = _first_present(bundle, ("preprocessor", "transformer"))
                    if transformer is not None and hasattr(transformer, "transform"):
                        Xt = transformer.transform(Xt)

                pca = bundle.get("pca")
                if pca is not None and hasattr(pca, "transform"):
                    Xt = pca.transform(Xt)

                expected_kmeans_dim = getattr(model, "n_features_in_", None)
                actual_dim = getattr(Xt, "shape", (None, None))[1]
                if expected_kmeans_dim is not None and actual_dim != expected_kmeans_dim:
                    raise ArtifactError(
                        "PD Cluster artifact tidak membawa preprocessing lengkap. "
                        f"KMeans mengharapkan {expected_kmeans_dim} input PCA features, "
                        f"tetapi runtime menghasilkan {actual_dim}. Re-export bundle dengan "
                        "risk_features + clip_bounds + imputer + scaler + pca + kmeans. "
                        "Missing borrower values tetap boleh NaN setelah bundle lengkap."
                    )
                cluster = model.predict(Xt)[0]

        try:
            cluster_out: Any = int(cluster)
        except Exception:
            cluster_out = str(cluster)
        return {
            "model": key,
            "task": "clustering",
            "cluster": cluster_out,
            "cluster_profile": self._cluster_profile(key, cluster_out),
            "artifact_info": self._metadata_excerpt(self.metadata(key), self.manifest(key)),
            "input_quality": self.input_quality(key, features),
            "input_reference_warnings": self._input_reference_warnings(key, features),
        }

    def predict(self, key: str, features: dict[str, Any]) -> dict[str, Any]:
        task = spec_for(key).task
        if task == "classification":
            return self.predict_classifier(key, features)
        if task == "regression":
            return self.predict_regression(key, features)
        if task == "clustering":
            return self.predict_cluster(key, features)
        raise ArtifactError(f"Task {task!r} belum didukung.")
