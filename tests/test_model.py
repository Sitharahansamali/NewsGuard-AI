import mlflow
import mlflow.sklearn


MODEL_URI = (
    "models:/NewsGuard-Fake-News-Classifier@champion"
)


def test_champion_model_loads():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    assert model is not None


def test_champion_model_predicts():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    text = [
        "Scientists announced a new scientific discovery."
    ]

    prediction = model.predict(text)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_champion_model_probability():

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    model = mlflow.sklearn.load_model(
        MODEL_URI
    )

    text = [
        "Scientists announced a new scientific discovery."
    ]

    probabilities = model.predict_proba(text)

    assert probabilities.shape == (1, 2)

    assert 0 <= probabilities[0][0] <= 1
    assert 0 <= probabilities[0][1] <= 1

    assert abs(
        sum(probabilities[0]) - 1
    ) < 0.0001