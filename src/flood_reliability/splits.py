"""Check process-level isolation before using any experimental split."""

from collections.abc import Iterable, Mapping
from itertools import combinations


def _valid_identifier(value: object) -> bool:
    """Allow only visible, whitespace-free ASCII machine identifiers."""
    return (
        isinstance(value, str)
        and bool(value)
        and value.isascii()
        and value.isprintable()
        and not any(char.isspace() for char in value)
    )


def canonical_process_id(value: object) -> str:
    """Validate and normalize the process key shared by splits and metrics."""
    if not _valid_identifier(value):
        raise ValueError("Process ID must be visible ASCII without whitespace")
    return value.casefold()


def process_overlap_report(groups: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    """Return shared process IDs, rejecting ambiguous or malformed identifiers."""
    normalized: dict[str, dict[str, str]] = {}
    for name, processes in groups.items():
        if not _valid_identifier(name):
            raise ValueError("Each split needs a visible ASCII name without whitespace")
        if isinstance(processes, (str, bytes)):
            raise ValueError(f"Split {name} needs an iterable of process IDs, not a scalar string")
        canonical = {}
        for process in processes:
            canonical[canonical_process_id(process)] = process
        if not canonical:
            raise ValueError(f"Split {name} needs at least one process ID")
        normalized[name] = canonical
    if len(normalized) < 2:
        raise ValueError("At least two splits are required")
    return {
        f"{left}__{right}": sorted(normalized[left][key] for key in normalized[left].keys() & normalized[right].keys())
        for left, right in combinations(normalized, 2)
        if normalized[left].keys() & normalized[right].keys()
    }


def require_process_disjoint(groups: Mapping[str, Iterable[str]]) -> None:
    """Refuse a split with any process present in more than one partition."""
    overlaps = process_overlap_report(groups)
    if overlaps:
        summary = ", ".join(f"{pair}: {len(ids)}" for pair, ids in overlaps.items())
        raise ValueError(f"Flood processes cross split boundaries ({summary})")
