"""Chronological forecasting benchmark and optional Optuna tuning with MLflow logging."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error

from backend import engine

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "model_eval"


def metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    actual_exceed = y_true > threshold
    predicted_exceed = y_pred > threshold
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "exceedance_f1": float(f1_score(actual_exceed, predicted_exceed, zero_division=0)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="PM25")
    parser.add_argument("--city", default="beijing")
    parser.add_argument("--trials", type=int, default=5)
    args = parser.parse_args()
    df = engine.preprocess(engine.load_data())
    data = engine.get_city_df(df, args.city)
    features = engine.get_features(args.target)
    train_end, val_end = int(len(data) * 0.70), int(len(data) * 0.85)
    train, validation, test = data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
    models = {
        "random_forest_baseline": RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }
    results: dict[str, dict[str, float]] = {}
    params: dict[str, dict] = {}
    for name, model in models.items():
        model.fit(train[features], train[args.target])
        results[name] = metrics(
            test[args.target].to_numpy(),
            model.predict(test[features]),
            engine.THRESHOLDS[args.target],
        )
        params[name] = model.get_params()
    best_name = min(results, key=lambda key: results[key]["rmse"])
    if args.trials > 0:
        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                model = ExtraTreesRegressor(
                    n_estimators=trial.suggest_int("n_estimators", 100, 400),
                    max_depth=trial.suggest_int("max_depth", 6, 24),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", 0.5, 1.0]),
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(train[features], train[args.target])
                return float(np.sqrt(mean_squared_error(
                    validation[args.target], model.predict(validation[features])
                )))

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=args.trials)
            tuned = ExtraTreesRegressor(**study.best_params, random_state=42, n_jobs=-1)
            tuned.fit(train[features], train[args.target])
            results["extra_trees_optuna"] = metrics(
                test[args.target].to_numpy(), tuned.predict(test[features]), engine.THRESHOLDS[args.target]
            )
            params["extra_trees_optuna"] = study.best_params
            best_name = min(results, key=lambda key: results[key]["rmse"])
        except ImportError:
            pass
    baseline = results["random_forest_baseline"]
    promoted = (
        best_name != "random_forest_baseline"
        and results[best_name]["rmse"] < baseline["rmse"]
        and results[best_name]["exceedance_f1"] >= baseline["exceedance_f1"]
    )
    result = {
        "created_at": datetime.now(UTC).isoformat(),
        "city": args.city,
        "target": args.target,
        "split": "70/15/15 chronological",
        "results": results,
        "best_model": best_name,
        "promoted": promoted,
        "best_params": params[best_name],
        "promotion_rule": "test RMSE lower and exceedance F1 not lower than baseline",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT / "forecast_tuning.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import mlflow

        mlflow.set_experiment("air-quality-forecast")
        with mlflow.start_run(run_name=f"{args.city}-{args.target}-tuning"):
            mlflow.log_params({"city": args.city, "target": args.target, "trials": args.trials})
            for name, model_metrics in results.items():
                mlflow.log_metrics({f"{name}_{key}": value for key, value in model_metrics.items()})
            mlflow.log_artifact(str(output))
    except ImportError:
        pass
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
