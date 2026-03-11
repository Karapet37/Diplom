from __future__ import annotations

import json
from pathlib import Path

import pytest

from style_tests.style_metrics import dump_metrics_json, evaluate_synthetic_suite


STYLE_TEST_FILES = [
    "style_tests/style_learning_tests.py",
    "style_tests/style_embedding_tests.py",
    "style_tests/style_generation_tests.py",
    "style_tests/style_isolation_tests.py",
]


def main() -> int:
    exit_code = pytest.main(STYLE_TEST_FILES)
    metrics = evaluate_synthetic_suite()
    output_path = Path("style_tests") / "style_test_metrics.json"
    dump_metrics_json(output_path, metrics)
    summary = {
        "pytest_exit_code": exit_code,
        "metrics_file": str(output_path),
        "style_similarity_score": metrics["avg_style_similarity_score"],
        "semantic_preservation_score": metrics["avg_semantic_preservation_score"],
        "style_drift_score": metrics["avg_style_drift_score"],
        "debug_logs": metrics["debug_logs"][:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
