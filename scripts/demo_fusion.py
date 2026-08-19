import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from app.config import Settings  # noqa: E402
from app.pipeline import AnalysisPipeline  # noqa: E402
from app.schemas import AnalyzeRequest  # noqa: E402


def main() -> None:
    result = AnalysisPipeline(Settings()).analyze(AnalyzeRequest(sample_id="synthetic-default"))
    figure, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(result.timestamps, result.aligned_signals["heart_rate"], label="Heart rate")
    axes[0].plot(result.timestamps, result.aligned_signals["spo2"], label="SpO2")
    axes[0].legend()
    axes[1].plot(result.timestamps, result.aligned_signals["respiration_rate"], label="Respiration")
    axes[1].plot(result.timestamps, result.aligned_signals["temperature"], label="Temperature")
    axes[1].legend()
    axes[2].plot(result.timestamps, result.fused_state, color="teal", label="Smoothed fused state")
    axes[2].fill_between(
        result.timestamps,
        [a - b for a, b in zip(result.fused_state, result.uncertainty, strict=True)],
        [a + b for a, b in zip(result.fused_state, result.uncertainty, strict=True)],
        alpha=0.2,
    )
    for event in result.anomalies:
        axes[2].axvspan(event.start_time, event.end_time, color="crimson", alpha=0.08)
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    figure.suptitle("Multimodal vital-sign fusion and detected events")
    figure.tight_layout()
    output = Path("artifacts/fusion_demo.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
