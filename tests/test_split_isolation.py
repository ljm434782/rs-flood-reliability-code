import pytest

from flood_reliability.splits import process_overlap_report, require_process_disjoint


def test_process_overlap_blocks_reused_event_even_with_distinct_tiles():
    groups = {"train": ["EMSR100", "EMSR200"], "validation": ["EMSR200"], "test": ["EMSR300"]}
    assert process_overlap_report(groups) == {"train__validation": ["EMSR200"]}
    with pytest.raises(ValueError, match="train__validation: 1"):
        require_process_disjoint(groups)


def test_disjoint_processes_pass_and_empty_group_is_rejected():
    require_process_disjoint({"train": ["EMSR100"], "validation": ["EMSR200"], "calibration": ["EMSR300"], "test": ["EMSR400"]})
    with pytest.raises(ValueError, match="at least one"):
        require_process_disjoint({"train": ["EMSR100"], "test": []})
    with pytest.raises(ValueError, match="At least two"):
        require_process_disjoint({"train": ["EMSR100"]})


def test_case_variant_process_id_cannot_cross_splits():
    groups = {"train": ["EMSR388"], "test": ["emsr388"]}
    assert process_overlap_report(groups) == {"train__test": ["EMSR388"]}
    with pytest.raises(ValueError, match="cross split boundaries"):
        require_process_disjoint(groups)


@pytest.mark.parametrize("groups", [
    {"train": [" EMSR388"], "test": ["EMSR200"]},
    {"train": ["EMSR388"], "test": [" "]},
    {"train": "EMSR388", "test": ["EMSR200"]},
    {"train ": ["EMSR388"], "test": ["EMSR200"]},
    {"train": ["EMSR388\u200b"], "test": ["EMSR388"]},
    {"train": ["ＥＭＳＲ３８８"], "test": ["EMSR388"]},
    {"train": ["EMSR\n388"], "test": ["EMSR388"]},
    {"train": ["EMSR 388"], "test": ["EMSR388"]},
])
def test_malformed_identifiers_fail_closed(groups):
    with pytest.raises(ValueError):
        require_process_disjoint(groups)
