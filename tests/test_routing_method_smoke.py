import os
from pathlib import Path
from unittest import mock

import pytest

from scripts.smoke_routing_methods import (
    cost_analysis,
    ensure_no_provider_credentials,
    parse_dataset_registry,
)


def test_provider_credentials_fail_closed_without_exposing_value() -> None:
    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "do-not-print"}, clear=True):
        with pytest.raises(RuntimeError) as caught:
            ensure_no_provider_credentials()
    assert "OPENAI_API_KEY" in str(caught.value)
    assert "do-not-print" not in str(caught.value)


def test_cost_analysis_detects_positive_row_denominator() -> None:
    items = [{"cost": 1.0}, {"cost": 2.0}, {"cost": 0.0}]
    summary = {
        "total_cost": 3.0,
        "total_items": 3,
        "avg_cost_per_item": 1.5,
    }
    result = cost_analysis(items, summary)
    assert result["total_cost_matches_items"] is True
    assert result["positive_cost_items"] == 2
    assert result["inferred_stored_divisor"] == 2.0
    assert result["stored_average_uses_all_items"] is False


def test_dataset_registry_parser_has_explicit_boolean_contract(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        "datasets:\n  ready:\n    implemented: true\n  blocked:\n    implemented: false\n",
        encoding="utf-8",
    )
    assert parse_dataset_registry(registry) == {"ready": True, "blocked": False}
