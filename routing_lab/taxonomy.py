"""Bounded, versioned route catalogs shared by clients and the Azure scorer."""

from collections.abc import Iterable
from types import MappingProxyType
from typing import Literal

TaxonomyId = Literal["coding-v1", "copilot-coding-v2", "tool-api-v1", "capability-v1"]
TAXONOMIES: tuple[TaxonomyId, ...] = ("coding-v1", "copilot-coding-v2", "tool-api-v1", "capability-v1")
# The kinds of work a request can involve: an optional second question next to the capability tier.
SKILLS: tuple[str, ...] = ("coding", "reasoning", "tools", "writing", "extraction")
_ROUTES = MappingProxyType({
    "coding-v1": (
        "debugging", "code_generation", "test_generation",
        "code_review", "repository_qa", "fallback",
    ),
    "copilot-coding-v2": (
        "debugging", "code_generation", "refactoring", "test_generation",
        "code_review", "security_review", "performance_optimization", "repository_qa",
        "research", "planning", "documentation", "build_release", "clarification", "fallback",
    ),
    "tool-api-v1": (
        "tool_lookup", "tool_action", "tool_workflow", "web_research",
        "memory_retrieval", "clarification", "no_matching_tool",
    ),
    # Ordered by capability: the cheapest model class that would do the job well.
    "capability-v1": ("low", "medium", "high"),
})


def taxonomy_routes(taxonomy_id: str) -> tuple[str, ...]:
    """Return the ordered routes, rejecting unknown taxonomy versions."""
    if taxonomy_id not in _ROUTES:
        raise ValueError(f"Unknown taxonomy: {taxonomy_id!r}")
    return _ROUTES[taxonomy_id]


def taxonomy_for_routes(routes: Iterable[str], taxonomy_id: str | None = None) -> TaxonomyId:
    """Resolve only complete exact catalogs; never infer from a subset of labels."""
    keys = tuple(routes)
    if taxonomy_id is not None:
        taxonomy_routes(taxonomy_id)
    for candidate in TAXONOMIES:
        expected = taxonomy_routes(candidate)
        if len(keys) == len(expected) and set(keys) == set(expected):
            if taxonomy_id is not None and candidate != taxonomy_id:
                raise ValueError(f"Catalog taxonomy {candidate} does not match {taxonomy_id}")
            return candidate
    raise ValueError("Catalog must contain exactly the routes of one supported taxonomy")


def abstention_routes(taxonomy_id: str) -> tuple[str, ...]:
    """Routes that defer ownership rather than execute a specialized task."""
    return tuple(route for route in taxonomy_routes(taxonomy_id)
                 if route in {"fallback", "clarification", "no_matching_tool"})
