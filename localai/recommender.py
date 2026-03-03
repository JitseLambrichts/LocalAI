"""
Recommendation engine.

Matches hardware capabilities to compatible Ollama models and ranks them.
"""

from dataclasses import dataclass

from localai.hardware import HardwareInfo
from localai.models import OllamaModel, get_all_models


@dataclass
class Recommendation:
    """A model recommendation with performance context."""
    model: OllamaModel
    performance: str          # "fast", "medium", "slow"
    performance_icon: str     # 🟢, 🟡, 🔴
    headroom_pct: float       # How much memory headroom (0-100)
    will_use_gpu: bool        # Whether GPU acceleration will be used
    note: str = ""            # Optional user-facing note


def _classify_performance(
    model: OllamaModel,
    effective_vram: float,
    has_gpu: bool,
) -> tuple[str, str, float]:
    """
    Classify expected performance for a model on given hardware.

    Returns (label, icon, headroom_percentage).
    """
    headroom = ((effective_vram - model.vram_gb) / effective_vram) * 100 if effective_vram > 0 else 0

    if not has_gpu:
        return "slow", "🔴", headroom

    if headroom >= 30:
        return "fast", "🟢", round(headroom, 1)
    elif headroom >= 10:
        return "medium", "🟡", round(headroom, 1)
    else:
        return "slow", "🔴", round(headroom, 1)


def get_recommendations(hardware: HardwareInfo) -> list[Recommendation]:
    """
    Get all compatible model recommendations for the detected hardware.

    Models are sorted by:
    1. Quality tier (descending — best first)
    2. Parameters (descending — larger is better within same tier)

    Models that don't fit in memory or disk are filtered out.
    """
    models = get_all_models()
    effective_vram = hardware.effective_vram_gb
    recommendations = []

    for model in models:
        # Check if model fits in available memory
        if model.vram_gb > effective_vram:
            continue

        # Check if we have enough disk space
        if model.size_gb > hardware.disk_available_gb:
            continue

        performance, icon, headroom = _classify_performance(
            model, effective_vram, hardware.has_gpu,
        )

        # Generate contextual notes
        note = ""
        if not hardware.has_gpu:
            note = "Will run on CPU — expect slower speeds"
        elif performance == "fast":
            note = "Plenty of headroom — will run smoothly"
        elif performance == "slow" and hardware.has_gpu:
            note = "Tight fit — may be slow with long prompts"

        recommendations.append(Recommendation(
            model=model,
            performance=performance,
            performance_icon=icon,
            headroom_pct=headroom,
            will_use_gpu=hardware.has_gpu,
            note=note,
        ))

    # Sort: quality tier desc, then parameters desc
    recommendations.sort(
        key=lambda r: (r.model.quality_tier, r.model.parameters),
        reverse=True,
    )

    return recommendations


def get_top_pick(recommendations: list[Recommendation]) -> Recommendation | None:
    """
    Get the single best model recommendation.

    Picks the highest quality model that still runs at 'fast' or 'medium' speed.
    Falls back to the highest quality model regardless.
    """
    if not recommendations:
        return None

    # Prefer models that run fast or medium
    comfortable = [r for r in recommendations if r.performance in ("fast", "medium")]
    if comfortable:
        return comfortable[0]

    # Fall back to first recommendation (highest quality)
    return recommendations[0]


def get_recommendations_by_category(
    recommendations: list[Recommendation],
) -> dict[str, list[Recommendation]]:
    """
    Group recommendations by capability category.

    Returns a dict like:
    {
        "chat": [...],
        "code": [...],
        "vision": [...],
        ...
    }
    """
    categories: dict[str, list[Recommendation]] = {}

    for rec in recommendations:
        for cap in rec.model.capabilities:
            if cap not in categories:
                categories[cap] = []
            categories[cap].append(rec)

    return categories
