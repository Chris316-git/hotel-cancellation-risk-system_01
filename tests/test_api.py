import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from hotel_risk import api  # noqa: E402
from hotel_risk.predict import save_bundle  # noqa: E402
from hotel_risk.train import build_bundle  # noqa: E402
from hotel_risk.data import clean  # noqa: E402
from tests.test_predict import synthetic  # noqa: E402


def test_health_and_score(tmp_path, monkeypatch):
    df = synthetic()
    full = df.assign(assigned_room_type="A", booking_changes=0, days_in_waiting_list=0,
                     required_car_parking_spaces=0, total_of_special_requests=0,
                     reservation_status="x", reservation_status_date="2017-01-01")
    path = tmp_path / "m.joblib"
    save_bundle(build_bundle(clean(full)), path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    api._bundle = None
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    rec = df.drop(columns="is_canceled").head(1).to_dict("records")
    rec[0]["agent"] = None if rec[0]["agent"] != rec[0]["agent"] else rec[0]["agent"]
    rec[0]["country"] = rec[0]["country"] or None
    rec[0]["company"] = None
    r = c.post("/score", json=rec)
    assert r.status_code == 200, r.text
    assert 0 <= r.json()[0]["cancel_probability"] <= 1
