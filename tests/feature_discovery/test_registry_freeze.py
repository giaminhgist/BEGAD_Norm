from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from begad_norm.feature_discovery import candidate_freeze as cf
from begad_norm.feature_discovery.dataset_features import d_table, validate_d_definitions
from begad_norm.feature_discovery.registry import BASE_CSV, base_registry, combined_registry, parse_base_catalogue
from begad_norm.feature_discovery.statistics import detect_aliases

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "Running_Phase/BEGAD_feature_selection_pack/01_FEATURE_CATALOG_160.md"


def test_base_registry_preserves_exact_160_ids():
    b = base_registry()
    assert len(b) == 160
    assert set(b.feature_id) == {f"M{i:03d}" for i in range(1, 81)} | {f"A{i:03d}" for i in range(1, 81)}
    assert (b.level == "micro").sum() == 80 and (b.level == "macro").sum() == 80
    assert b.status.isin(["supported", "blocked_source", "blocked_semantics"]).all()
    assert (b.loc[b.status != "supported", "status_reason"] != "").all()


@pytest.mark.skipif(not PACK.exists(), reason="instruction pack not present")
def test_committed_base_catalogue_matches_instruction_pack_text():
    parsed = parse_base_catalogue(PACK)
    committed = pd.read_csv(BASE_CSV, dtype=str, keep_default_na=False)
    pd.testing.assert_frame_equal(parsed.reset_index(drop=True).astype(str), committed.astype(str))


def test_combined_registry_partial_extension_counts_and_explicit_levels():
    reg = combined_registry()
    d = reg[reg.origin == "dataset100"]
    assert 0 < len(d) <= 100
    assert list(d.feature_id) == [f"D{i:03d}" for i in range(1, len(d) + 1)]
    assert len(reg) == 160 + len(d)
    assert set(d.level) == {"micro", "macro"}
    assert not reg.feature_id.duplicated().any()


def test_d_validation_rejects_placeholders_and_duplicates():
    d = d_table()
    bad = d.copy()
    bad.loc[0, "formula"] = ""
    with pytest.raises(ValueError):
        validate_d_definitions(bad)
    dup = pd.concat([d, d.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_d_definitions(dup)
    renamed = d.copy()
    renamed.loc[1, "name"] = renamed.loc[0, "name"]
    with pytest.raises(ValueError):
        validate_d_definitions(renamed)


def test_rank_alias_detection_catches_reciprocal_and_unit_change_not_difference():
    rng = np.random.default_rng(1)
    x = rng.uniform(1, 5, 30)
    start = rng.uniform(2, 4, 30)
    end = start + rng.normal(0, .3, 30)
    rows = {("C", "D005"): x, ("C", "X_recip"): 1 / x, ("C", "X_ms"): x * 1000,
            ("C", "P_start"): start, ("C", "P_end"): end, ("C", "P_diff"): end - start}
    mat = pd.DataFrame(rows).T
    mat.index = pd.MultiIndex.from_tuples(mat.index)
    al = detect_aliases(mat, priority=lambda f: (0, f))
    pairs = {tuple(sorted((a, b))) for a, b in zip(al.representative, al.alias)}
    assert ("D005", "X_recip") in pairs and ("D005", "X_ms") in pairs
    assert not any("P_diff" in p for p in pairs)


def test_freeze_detects_changed_configuration(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text((ROOT / "configs/feature_discovery.yaml").read_text())
    cf.write_freeze(tmp_path, cfg, {"items": []})
    cf.verify_freeze(tmp_path, cfg)
    cfg.write_text(cfg.read_text().replace("screen_p: 0.20", "screen_p: 0.25"))
    with pytest.raises(RuntimeError):
        cf.verify_freeze(tmp_path, cfg)
