from app.config import Settings
from app.pipeline import AnalysisPipeline
from app.schemas import AnalyzeRequest


def test_synthetic_window_detects_expected_event_types():
    result = AnalysisPipeline(Settings()).analyze(AnalyzeRequest(sample_id="synthetic-default"))
    kinds = {event.anomaly_type for event in result.anomalies}
    assert "oxygen_desaturation" in kinds
    assert "heart_rate_irregularity" in kinds
    assert "temperature_elevation" in kinds
    assert all(0 <= event.confidence <= 1 for event in result.anomalies)
