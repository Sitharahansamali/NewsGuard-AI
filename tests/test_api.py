from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == (
        "Fake News Detection API Running"
    )

    assert data["model"] == (
        "NewsGuard-Fake-News-Classifier"
    )

    assert data["model_alias"] == "champion"


def test_predict():

    response = client.post(
        "/predict",
        params={
            "text": (
                "Scientists announced "
                "a new discovery."
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "confidence" in data

    assert data["prediction"] in [
        "Real News",
        "Fake News",
    ]

    assert 0 <= data["confidence"] <= 1