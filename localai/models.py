"""
Ollama model database — fetched live from the Ollama library.

Scrapes the official model listing at https://ollama.com/search to get
all available models with their capabilities and size variants, then
estimates hardware requirements for each.
"""

import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass, field

OLLAMA_SEARCH_URL = "https://ollama.com/search"
OLLAMA_API_TAGS_URL = "https://ollama.com/api/tags"
FETCH_TIMEOUT = 15  # seconds


@dataclass
class OllamaModel:
    """An Ollama model with its requirements and metadata."""
    name: str               # Model name in Ollama (e.g., "llama3.2")
    variant: str            # Size variant (e.g., "3b", "7b")
    parameters: float       # Billions of parameters
    size_gb: float          # Download size in GB (quantized)
    vram_gb: float          # Minimum VRAM/memory needed to run
    capabilities: list = field(default_factory=list)  # ["chat", "code", ...]
    quality_tier: int = 3   # 1 (basic) to 5 (state-of-the-art)
    description: str = ""   # User-friendly one-liner

    @property
    def full_name(self) -> str:
        """Return the full model name for `ollama pull`."""
        return f"{self.name}:{self.variant}"

    @property
    def param_label(self) -> str:
        """Human-friendly parameter count."""
        if self.parameters >= 1:
            return f"{self.parameters:.0f}B" if self.parameters == int(self.parameters) else f"{self.parameters}B"
        return f"{self.parameters * 1000:.0f}M"

    @property
    def capability_icons(self) -> str:
        """Return emoji icons for capabilities."""
        icons = {
            "chat": "💬",
            "code": "💻",
            "vision": "👁️",
            "reasoning": "🧠",
            "rag": "📚",
            "math": "🔢",
            "embedding": "🔗",
            "tools": "🔧",
        }
        return " ".join(icons.get(c, "•") for c in self.capabilities)


# ─── Helper Functions ────────────────────────────────────────────────

def _parse_parameters(variant: str) -> float:
    """
    Parse parameter count from a variant tag string.

    Examples:
        "8b"   → 8.0
        "3.8b" → 3.8
        "235b" → 235.0
        "1t"   → 1000.0
        "0.6b" → 0.6
        "latest" → 0.0 (unknown)
    """
    variant = variant.lower().strip()

    match = re.match(r"^(\d+(?:\.\d+)?)b$", variant)
    if match:
        return float(match.group(1))

    match = re.match(r"^(\d+(?:\.\d+)?)t$", variant)
    if match:
        return float(match.group(1)) * 1000

    return 0.0


def _estimate_size_gb(parameters: float) -> float:
    """
    Estimate download size in GB from parameter count.

    Rule of thumb for Q4_K_M quantization:
    ~0.5-0.6 GB per billion parameters.
    """
    if parameters <= 0:
        return 0.0
    return round(parameters * 0.55, 1)


def _estimate_vram(size_gb: float) -> float:
    """
    Estimate minimum VRAM needed from the download size.

    At runtime, Ollama needs extra memory for the KV cache,
    compute buffers, and overhead (~25%).
    """
    return round(size_gb * 1.25, 1)


def _parse_capabilities(cap_tags: list[str], model_name: str) -> list[str]:
    """
    Build a capability list from HTML tags and model name.

    cap_tags come from the search page (e.g. ['vision', 'tools', 'thinking']).
    Additional capabilities are inferred from the model name.
    """
    caps = []
    name_lower = model_name.lower()
    tag_set = {t.lower() for t in cap_tags}

    # Vision
    if "vision" in tag_set or any(kw in name_lower for kw in ["vl", "vision", "llava", "ocr"]):
        caps.append("vision")

    # Code
    if any(kw in name_lower for kw in ["code", "coder", "starcoder", "devstral"]):
        caps.append("code")

    # Reasoning / thinking
    if "thinking" in tag_set or any(kw in name_lower for kw in ["r1", "r2", "thinking", "reason", "cogito"]):
        caps.append("reasoning")

    # Tools / function calling
    if "tools" in tag_set:
        caps.append("tools")

    # Embedding
    if "embedding" in tag_set or "embed" in name_lower:
        caps.append("embedding")

    # RAG
    if any(kw in name_lower for kw in ["command-r", "rag"]):
        caps.append("rag")

    # Math
    if any(kw in name_lower for kw in ["math", "wizard-math"]):
        caps.append("math")

    # Default: everything is at least a chat model (unless embedding-only)
    if "embedding" not in caps:
        caps.insert(0, "chat")

    return caps


def _estimate_quality_tier(parameters: float) -> int:
    """
    Assign a quality tier (1-5) based on parameter count.
    """
    if parameters <= 0:
        return 3
    if parameters < 2:
        return 1
    if parameters < 5:
        return 2
    if parameters < 15:
        return 3
    if parameters < 40:
        return 4
    return 5


# ─── Search Page Scraper ─────────────────────────────────────────────

def _scrape_search_page(url: str) -> str:
    """Fetch a single search page and return the HTML."""
    req = urllib.request.Request(url, headers={"User-Agent": "LocalAI/1.0"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        return resp.read().decode("utf-8")


def _parse_model_cards(html: str) -> list[dict]:
    """
    Parse model cards from the Ollama search page HTML.

    Each model card is a list item with a link to /library/<name>
    containing: description, capability tags, and size variants.

    Returns a list of dicts with keys:
      name, description, capabilities, sizes
    """
    models = []

    # Split HTML into model blocks using the library links as delimiters
    # Each block starts with href="/library/<name>" and contains the card data
    blocks = re.split(r'(?=href="/library/)', html)

    for block in blocks:
        # Extract model name
        name_match = re.search(r'href="/library/([^"]+)"', block)
        if not name_match:
            continue
        model_name = name_match.group(1)

        # Skip duplicates within the same block (cards appear twice in the HTML)
        # and skip non-model paths like "/library/modelname/tags"
        if "/" in model_name:
            continue

        # Extract description by stripping HTML first, then finding text after model name
        desc = ""
        clean_text = re.sub(r'<[^>]+>', ' ', block)  # Strip all HTML tags
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        desc_match = re.search(
            rf'{re.escape(model_name)}\s+(.+?)(?:vision|tools|thinking|embedding|cloud|code|\d+(?:\.\d+)?b|\d+[\d,.]*[KM]\s*Pulls)',
            clean_text,
            re.IGNORECASE,
        )
        if desc_match:
            raw_desc = desc_match.group(1).strip().rstrip('.')
            # Remove leftover class names, CSS-like content
            raw_desc = re.sub(r'class=\S+', '', raw_desc)
            raw_desc = re.sub(r'["\']', '', raw_desc)
            raw_desc = re.sub(r'\s+', ' ', raw_desc).strip()
            if len(raw_desc) > 15 and not raw_desc.startswith('group'):
                desc = raw_desc

        # Extract capability tags (vision, tools, thinking, embedding, cloud, code)
        cap_tags = re.findall(
            r'>(vision|tools|thinking|embedding|code)<',
            block[:3000],
            re.IGNORECASE,
        )

        # Extract size variants (0.8b, 2b, 4b, 9b, 27b, etc.)
        sizes = re.findall(r'>(\d+(?:\.\d+)?b)<', block[:3000], re.IGNORECASE)

        # Deduplicate
        sizes = list(dict.fromkeys(sizes))
        cap_tags = list(dict.fromkeys(cap_tags))

        # If no sizes found, add "latest" as default
        if not sizes:
            sizes = ["latest"]

        models.append({
            "name": model_name,
            "description": desc,
            "capabilities": cap_tags,
            "sizes": sizes,
        })

    # Deduplicate by name (cards appear twice in the HTML)
    seen = set()
    unique = []
    for m in models:
        if m["name"] not in seen:
            seen.add(m["name"])
            unique.append(m)

    return unique


def _fetch_all_search_pages() -> list[dict]:
    """
    Fetch all pages of the Ollama search listing.

    Returns a deduplicated list of model card dicts from all pages.
    """
    all_models: dict[str, dict] = {}  # name -> card
    page = 1
    max_pages = 20  # Safety limit

    while page <= max_pages:
        url = f"{OLLAMA_SEARCH_URL}?p={page}" if page > 1 else OLLAMA_SEARCH_URL

        try:
            html = _scrape_search_page(url)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            break

        models = _parse_model_cards(html)
        if not models:
            break  # No more models — we've passed the last page

        for m in models:
            if m["name"] not in all_models:
                all_models[m["name"]] = m

        page += 1

    return list(all_models.values())


# ─── API Enrichment ──────────────────────────────────────────────────

def _fetch_api_sizes() -> dict[str, int]:
    """
    Fetch model sizes from the /api/tags endpoint.

    Returns a dict mapping "name:variant" → size_in_bytes.
    This provides real download sizes for models that appear in the API.
    """
    try:
        req = urllib.request.Request(
            OLLAMA_API_TAGS_URL,
            headers={"User-Agent": "LocalAI/1.0"},
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError):
        return {}

    sizes = {}
    for entry in data.get("models", []):
        model_name = entry.get("name", "") or entry.get("model", "")
        size_bytes = entry.get("size", 0)
        if model_name and size_bytes > 0:
            sizes[model_name.lower()] = size_bytes

    return sizes


# ─── Public Fetching ─────────────────────────────────────────────────

def fetch_models_from_registry() -> list[OllamaModel]:
    """
    Fetch all official models from the Ollama library.

    1. Scrapes ollama.com/search for model names, capabilities, and size variants
    2. Enriches with real download sizes from /api/tags where available
    3. Estimates sizes for variants not in the API

    Returns an empty list if the fetch fails.
    """
    try:
        model_cards = _fetch_all_search_pages()
    except Exception:
        return []

    if not model_cards:
        return []

    # Get real sizes from the API for enrichment
    api_sizes = _fetch_api_sizes()

    seen: set[str] = set()  # Track name:variant to prevent duplicates
    result: list[OllamaModel] = []

    for card in model_cards:
        name = card["name"]
        description = card.get("description", "")
        cap_tags = card.get("capabilities", [])

        for size_tag in card["sizes"]:
            variant = size_tag.lower()
            parameters = _parse_parameters(variant)

            # Skip "cloud" variants (these run on Ollama's servers, not locally)
            if variant in ("cloud", "latest"):
                # For "latest", check if we have API data
                api_key = f"{name}:latest".lower()
                alt_key = name.lower()
                if api_key not in api_sizes and alt_key not in api_sizes:
                    continue
                # Use API data for "latest"
                actual_bytes = api_sizes.get(api_key, api_sizes.get(alt_key, 0))
                if actual_bytes > 0:
                    size_gb = round(actual_bytes / (1024 ** 3), 1)
                    if parameters <= 0:
                        parameters = round(size_gb / 0.55, 1)
                    variant = "latest"
                else:
                    continue

            # Deduplicate by name:variant
            dedup_key = f"{name}:{variant}".lower()
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Try to get real size from API
            api_key = f"{name}:{variant}".lower()
            if api_key in api_sizes:
                size_gb = round(api_sizes[api_key] / (1024 ** 3), 1)
            elif variant != "latest":
                # Estimate from parameter count
                size_gb = _estimate_size_gb(parameters)

            if size_gb <= 0 and parameters <= 0:
                continue

            vram_gb = _estimate_vram(size_gb)
            capabilities = _parse_capabilities(cap_tags, name)
            quality_tier = _estimate_quality_tier(parameters)

            # Build description
            if not description:
                param_str = f"{parameters:.0f}B" if parameters >= 1 else f"{parameters * 1000:.0f}M"
                description_line = f"{name} ({param_str} parameters)"
            else:
                description_line = description

            result.append(OllamaModel(
                name=name,
                variant=variant,
                parameters=parameters,
                size_gb=size_gb,
                vram_gb=vram_gb,
                capabilities=capabilities,
                quality_tier=quality_tier,
                description=description_line,
            ))

    return result


# ─── Public API ──────────────────────────────────────────────────────

def get_all_models() -> list[OllamaModel]:
    """
    Return the full model list from the Ollama registry.

    Fetches live from ollama.com. Returns an empty list on failure.
    """
    return fetch_models_from_registry()


def get_models_by_capability(capability: str) -> list[OllamaModel]:
    """Filter models by a specific capability."""
    return [m for m in get_all_models() if capability in m.capabilities]
