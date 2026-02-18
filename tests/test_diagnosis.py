import pytest
from io import BytesIO


@pytest.mark.asyncio
async def test_predict_success(client):
    file_content = b"fake_image_bytes"
    files = {"file": ("lung.jpg", BytesIO(file_content), "image/jpeg")}

    response = await client.post("/api/v1/lungcheck/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "PNEUMONIA"
    assert "confidence" in data
    assert data["filename"] == "lung.jpg"


@pytest.mark.asyncio
async def test_history_is_working(client):
    response = await client.get("/api/v1/lungcheck/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.parametrize("limit, expected_count", [
    (1, 1),
    (5, 5),
    (0, 0),
])
async def test_history_pagination(client, limit, expected_count):
    response = await client.get(f"/api/v1/lungcheck/history?limit={limit}")
    assert len(response.json()) == expected_count