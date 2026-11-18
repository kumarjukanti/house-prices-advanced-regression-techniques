"""Record verified local house-price artifacts; optionally publish to MLflow."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import yaml


def digest(path, algorithm="md5"):
    checksum = hashlib.new(algorithm, usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id")
    parser.add_argument("--model-name", default="house-price-model")
    parser.add_argument("--model-version")
    parser.add_argument("--sync-mlflow", action="store_true")
    args = parser.parse_args()
    if args.sync_mlflow and not args.run_id:
        parser.error("--sync-mlflow requires --run-id")
    if args.model_version and not args.run_id:
        parser.error("--model-version requires --run-id")

    root = args.project_root.resolve()
    paths = {
        "train": ("features", "data/processed/train.parquet"),
        "validation": ("features", "data/processed/validation.parquet"),
        "test": ("features", "data/processed/test.parquet"),
        "model": ("train", "models/selected_house_price_model.joblib"),
        "selection": ("train", "artifacts/model_selection.json"),
    }
    # Evaluation is optional until the evaluate stage has been added and run.
    evaluation_path = root / "reports/metrics.json"
    if evaluation_path.is_file():
        paths["test_metrics"] = ("evaluate", "reports/metrics.json")

    lock = yaml.safe_load((root / "dvc.lock").read_text())["stages"]
    verified = {}
    for name, (stage_name, relative) in paths.items():
        # DVC records metrics in the lockfile's outs group, including
        # metrics configured with cache: false.
        stage = lock.get(stage_name, {})
        entry = next(
            (item for item in stage.get("outs", [])
             if item["path"] == relative),
            None,
        )
        if entry is None or "md5" not in entry:
            raise ValueError(f"Missing DVC MD5 entry: {relative}. Run dvc repro.")
        path = root / relative
        actual = digest(path)
        if actual != entry["md5"]:
            raise ValueError(f"DVC hash mismatch: {relative}. Run dvc repro.")
        verified[name] = {"path": relative, "md5": actual}

    selection = json.loads((root / paths["selection"][1]).read_text())
    evaluation = (
        json.loads(evaluation_path.read_text())
        if "test_metrics" in paths else None
    )
    if evaluation and evaluation.get("selected_model") not in (
        None, selection["selected_model"]
    ):
        raise ValueError("Evaluation and selection name different models")

    # Only deserialize a joblib artifact produced by trusted code.
    model_path = root / paths["model"][1]
    model = joblib.load(model_path)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Saved pipeline does not record its input feature names")
    features = list(model.feature_names_in_)
    if {"Id", "SalePrice"} & set(features):
        raise ValueError("Id and SalePrice must not be model input features")

    lineage = {
        "project": "house-prices-advanced-regression-techniques",
        "task": "regression",
        "target": "SalePrice",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": selection["selected_model"],
        "selection_metric": selection.get("selection_metric"),
        "validation_rmse_log_price": selection.get("validation_rmse_log_price"),
        "features": features,
        "verified_local_artifacts": verified,
        "model_sha256": digest(model_path, "sha256"),
        "dvc_lock_sha256": digest(root / "dvc.lock", "sha256"),
        "test_metrics": evaluation,
        "mlflow": None,
        "limitations": [
            "Local files match the current DVC lock; this does not prove historical execution.",
            "Feature names were read from the saved pipeline; no historical schema version is asserted.",
            "No deployment decision is made by this script.",
        ],
    }

    client = None
    if args.run_id:
        from mlflow import MlflowClient

        client = MlflowClient()
        run = client.get_run(args.run_id)
        if run.info.status != "FINISHED":
            raise ValueError("MLflow run must be FINISHED")
        # The training code must log this tag for its exact saved joblib file.
        # Merely supplying a run ID does not establish artifact equivalence.
        if run.data.tags.get("lineage.model_sha256") != lineage["model_sha256"]:
            raise ValueError(
                "Cannot link the run: its lineage.model_sha256 tag must match "
                "the joblib hash recorded during training."
            )
        lineage["mlflow"] = {
            "tracking_uri": client.tracking_uri,
            "run_id": args.run_id,
            "training_git_sha": run.data.tags.get("mlflow.source.git.commit"),
            "started_at_utc": datetime.fromtimestamp(
                run.info.start_time / 1000, tz=UTC
            ).isoformat(),
        }
        if args.model_version:
            version = client.get_model_version(args.model_name, args.model_version)
            if version.run_id != args.run_id:
                raise ValueError("Registered model version does not identify this run")
            lineage["mlflow"]["registered_model_name"] = args.model_name
            lineage["mlflow"]["registered_model_version"] = args.model_version
            lineage["limitations"].append(
                "Registry run association checked; registered model bytes were not compared."
            )

    output = root / "artifacts/model_lineage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lineage, indent=2, allow_nan=False) + "\n")
    print(f"Lineage saved to: {output}")

    if args.sync_mlflow:
        client.log_artifact(args.run_id, str(output), artifact_path="lineage")
        tags = {"lineage.project": lineage["project"], "lineage.task": "regression"}
        for key, value in tags.items():
            client.set_tag(args.run_id, key, value)
            if args.model_version:
                client.set_model_version_tag(
                    args.model_name, args.model_version, key, value
                )
        print("Published lineage artifact and tags to MLflow.")


if __name__ == "__main__":
    main()
