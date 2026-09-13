import mlflow
from mlflow import MlflowClient


MODEL_NAME = "NewsGuard-Fake-News-Classifier"
CHAMPION_ALIAS = "champion"

# Latest complete pipeline run
RUN_ID = "5048fb20897c4487a9fd4b2ac0d27b1c"


def main():

    mlflow.set_tracking_uri("sqlite:///./mlflow.db")

    client = MlflowClient()

    run = client.get_run(RUN_ID)

    model_name = run.data.params.get("model_name")
    f1_score = run.data.metrics.get("f1_score")

    print("Model:", model_name)
    print("F1 Score:", f1_score)
    print("Run ID:", RUN_ID)

    # Register the complete pipeline
    model_uri = f"runs:/{RUN_ID}/model"

    print("\nRegistering complete pipeline...")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    version = registered_model.version

    print(f"Registered model version: {version}")

    # Make the new complete pipeline the champion
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=CHAMPION_ALIAS,
        version=version,
    )

    print(
        f"Champion alias -> Version {version}"
    )

    print("\nMigration completed successfully!")


if __name__ == "__main__":
    main()