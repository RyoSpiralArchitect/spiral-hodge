import csv

import pytest

from scripts import export_hltd_prefix_null_charts as charts


def write_rows(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_calibration_sql_excludes_unmatched_and_other_scopes(tmp_path):
    path = tmp_path / "draws.csv"
    base = {"comparator": "v2_random", "scope": "all", "matching": "common", "paired_nodes": 4, "paired_prompts": 2}
    write_rows(path, [{**base, "control_cosine": .2}, {**base, "control_cosine": .4},
                     {**base, "control_cosine": .9, "matching": "per_draw"},
                     {**base, "control_cosine": .8, "scope": "supported"}])
    rows = charts.query_csv(path, "calibration_draws", (charts.BASE / "calibration_chart.sql").read_text())
    assert len(rows) == 1 and rows[0]["draws"] == 2
    assert rows[0]["mean_cosine"] == pytest.approx(.3)


def test_v2_sql_uses_nested_equal_prompt_not_flat_row_average(tmp_path):
    path = tmp_path / "treatments.csv"
    base = {"component": "coexact", "alpha": 1., "next_token_logprob_base": -100.}
    rows = []
    for prompt, prefix, deltas in [("a", 8, [1., 3.]), ("a", 16, [10., 14.]), ("b", 8, [30., 50.])]:
        for seed, delta in enumerate(deltas):
            rows.append({**base, "prompt_id": prompt, "prefix_length": prefix, "seed": seed,
                         "next_token_logprob_steered": -100. + delta})
    write_rows(path, rows)
    output = charts.query_csv(path, "v2_treatments", (charts.BASE / "v2_chart.sql").read_text())
    assert output[0]["mean_delta_logprob"] == 23.5
    assert output[0]["prompts"] == 2 and output[0]["seeds"] == 2


def test_chart_export_is_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(charts, "BASE", tmp_path)
    (tmp_path / "charts").mkdir()
    with pytest.raises(FileExistsError):
        charts.export()
