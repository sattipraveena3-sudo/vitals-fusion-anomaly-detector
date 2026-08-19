import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data.synthetic import generate_synthetic_vitals  # noqa: E402


def main() -> None:
    output = Path("data/sample_vitals.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_synthetic_vitals().to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
