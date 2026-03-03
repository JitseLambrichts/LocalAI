"""
Ollama model database.

A curated list of popular Ollama models with their hardware requirements,
capabilities, and quality ratings.
"""

from dataclasses import dataclass, field


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
        }
        return " ".join(icons.get(c, "•") for c in self.capabilities)


# ─── Curated Model Database ──────────────────────────────────────────
# All sizes assume Q4_K_M quantization (Ollama default).
# VRAM estimates include ~0.5-1 GB overhead for KV cache at default context.

MODEL_DATABASE: list[OllamaModel] = [
    # ── Tiny Models (< 2 GB VRAM) ────────────────────────────────────
    OllamaModel(
        name="tinyllama", variant="1.1b",
        parameters=1.1, size_gb=0.6, vram_gb=1.0,
        capabilities=["chat"],
        quality_tier=1,
        description="Ultra-light model for basic chat on minimal hardware",
    ),
    OllamaModel(
        name="llama3.2", variant="1b",
        parameters=1.0, size_gb=1.3, vram_gb=1.5,
        capabilities=["chat"],
        quality_tier=2,
        description="Compact Meta model, great for simple tasks",
    ),

    # ── Small Models (2-4 GB VRAM) ───────────────────────────────────
    OllamaModel(
        name="llama3.2", variant="3b",
        parameters=3.0, size_gb=2.0, vram_gb=3.0,
        capabilities=["chat", "code"],
        quality_tier=3,
        description="Good balance of speed and quality for everyday use",
    ),
    OllamaModel(
        name="phi4-mini", variant="3.8b",
        parameters=3.8, size_gb=2.5, vram_gb=3.5,
        capabilities=["chat", "code", "reasoning"],
        quality_tier=3,
        description="Microsoft's compact model, surprisingly smart for its size",
    ),
    OllamaModel(
        name="gemma3", variant="4b",
        parameters=4.0, size_gb=3.3, vram_gb=4.0,
        capabilities=["chat", "vision"],
        quality_tier=3,
        description="Google's small model with vision capabilities",
    ),

    # ── Medium Models (4-8 GB VRAM) ──────────────────────────────────
    OllamaModel(
        name="llama3.1", variant="8b",
        parameters=8.0, size_gb=4.9, vram_gb=6.0,
        capabilities=["chat", "code"],
        quality_tier=4,
        description="Meta's flagship 8B — the community favorite",
    ),
    OllamaModel(
        name="mistral", variant="7b",
        parameters=7.0, size_gb=4.1, vram_gb=6.0,
        capabilities=["chat", "code"],
        quality_tier=4,
        description="Fast and efficient European model, great for chat",
    ),
    OllamaModel(
        name="qwen3", variant="8b",
        parameters=8.0, size_gb=4.9, vram_gb=6.0,
        capabilities=["chat", "code", "math"],
        quality_tier=4,
        description="Alibaba's versatile model, excels at reasoning and math",
    ),
    OllamaModel(
        name="deepseek-r1", variant="8b",
        parameters=8.0, size_gb=4.9, vram_gb=6.0,
        capabilities=["chat", "reasoning", "code"],
        quality_tier=4,
        description="DeepSeek's reasoning model with chain-of-thought",
    ),
    OllamaModel(
        name="codellama", variant="7b",
        parameters=7.0, size_gb=3.8, vram_gb=6.0,
        capabilities=["code"],
        quality_tier=3,
        description="Specialized for code generation and completion",
    ),

    # ── Large Models (8-16 GB VRAM) ──────────────────────────────────
    OllamaModel(
        name="gemma3", variant="12b",
        parameters=12.0, size_gb=8.1, vram_gb=10.0,
        capabilities=["chat", "vision"],
        quality_tier=4,
        description="Google's mid-size model with strong vision capabilities",
    ),
    OllamaModel(
        name="llama3.2-vision", variant="11b",
        parameters=11.0, size_gb=7.9, vram_gb=10.0,
        capabilities=["chat", "vision"],
        quality_tier=4,
        description="Meta's vision-language model, understands images",
    ),
    OllamaModel(
        name="codellama", variant="13b",
        parameters=13.0, size_gb=7.4, vram_gb=10.0,
        capabilities=["code"],
        quality_tier=4,
        description="Larger code model for complex programming tasks",
    ),

    # ── XL Models (16-24 GB VRAM) ────────────────────────────────────
    OllamaModel(
        name="codellama", variant="34b",
        parameters=34.0, size_gb=19.0, vram_gb=22.0,
        capabilities=["code"],
        quality_tier=4,
        description="Top-tier code model for professional development",
    ),
    OllamaModel(
        name="gemma3", variant="27b",
        parameters=27.0, size_gb=17.0, vram_gb=20.0,
        capabilities=["chat", "vision"],
        quality_tier=5,
        description="Google's high-quality model, rivals much larger models",
    ),
    OllamaModel(
        name="qwen3", variant="32b",
        parameters=32.0, size_gb=20.0, vram_gb=24.0,
        capabilities=["chat", "code", "math", "reasoning"],
        quality_tier=5,
        description="Alibaba's powerhouse, excellent across all tasks",
    ),
    OllamaModel(
        name="command-r", variant="35b",
        parameters=35.0, size_gb=20.0, vram_gb=24.0,
        capabilities=["chat", "rag"],
        quality_tier=4,
        description="Cohere's model optimized for RAG workflows",
    ),

    # ── XXL Models (48+ GB VRAM) ─────────────────────────────────────
    OllamaModel(
        name="llama3.1", variant="70b",
        parameters=70.0, size_gb=43.0, vram_gb=48.0,
        capabilities=["chat", "code"],
        quality_tier=5,
        description="Meta's largest open model — near-GPT-4 quality",
    ),
    OllamaModel(
        name="deepseek-r1", variant="70b",
        parameters=70.0, size_gb=43.0, vram_gb=48.0,
        capabilities=["chat", "reasoning", "code"],
        quality_tier=5,
        description="DeepSeek's large reasoning model, exceptional quality",
    ),
    OllamaModel(
        name="qwen3", variant="235b",
        parameters=235.0, size_gb=142.0, vram_gb=160.0,
        capabilities=["chat", "code", "math", "reasoning"],
        quality_tier=5,
        description="Alibaba's flagship — truly massive, datacenter-grade",
    ),
]


def get_all_models() -> list[OllamaModel]:
    """Return the full model database."""
    return MODEL_DATABASE.copy()


def get_models_by_capability(capability: str) -> list[OllamaModel]:
    """Filter models by a specific capability."""
    return [m for m in MODEL_DATABASE if capability in m.capabilities]
