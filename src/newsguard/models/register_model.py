import mlflow
from mlflow import MlflowClient


EXPERIMENT_NAME = "NewsGuard AI - Fake News Classification"
MODEL_NAME = "NewsGuard-Fake-News-Classifier"


def main():
    # Connect to the same MLflow database
    mlflow.set_tracking_uri("sqlite:///./mlflow.db")

    client = MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found.")

    # Find the best run based on F1 score
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found.")

    best_run = runs[0]

    run_id = best_run.info.run_id
    f1_score = best_run.data.metrics["f1_score"]
    model_name = best_run.data.params["model_name"]

    print(f"Best model: {model_name}")
    print(f"Run ID: {run_id}")
    print(f"F1 Score: {f1_score:.6f}")

    # Model URI inside MLflow
    model_uri = f"runs:/{run_id}/model"

    # Register model
    result = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    print("\nModel registered successfully!")
    print(f"Model name: {result.name}")
    print(f"Model version: {result.version}")


    # Set the champion alias
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=result.version,
    )

    print(f"Alias 'champion' assigned to version {result.version}")


if __name__ == "__main__":
    main()