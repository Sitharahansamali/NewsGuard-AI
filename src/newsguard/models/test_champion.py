import mlflow


MODEL_NAME = "NewsGuard-Fake-News-Classifier"
ALIAS = "champion"


def main():

    mlflow.set_tracking_uri("sqlite:///./mlflow.db")

    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

    print("Loading:")
    print(model_uri)

    model = mlflow.pyfunc.load_model(model_uri)

    print("\nChampion model loaded successfully!")
    print("Model type:", type(model))

    # Test raw news text
    sample_text = """
    Scientists have announced a new breakthrough in renewable
    energy technology that could significantly reduce the cost
    of solar power.
    """

    prediction = model.predict([sample_text])

    print("\nPrediction:")
    print(prediction)


if __name__ == "__main__":
    main()