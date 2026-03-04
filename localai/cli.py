"""
CLI interface — the main entry point for LocalAI.

Beautiful, user-friendly terminal output with rich formatting.
"""

import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.padding import Padding
from rich import box

from localai.hardware import detect_hardware, HardwareInfo
from localai.models import OllamaModel, get_all_models
from localai.recommender import (
    get_recommendations,
    get_top_pick,
    get_recommendations_by_category,
    Recommendation,
    _classify_performance,
)
from localai.ollama import (
    is_ollama_installed,
    get_ollama_version,
    get_installed_models,
)

console = Console()

# ─── Brand / Styling ─────────────────────────────────────────────────

LOGO = r"""
  _                     _    _    ___
 | |    ___   ___ __ _ | |  / \  |_ _|
 | |   / _ \ / __/ _` || | / ^ \  | |
 | |__| (_) | (_| (_| || |/ ___ \ | |
 |_____\___/ \___\__,_||_/_/   \_\___|
"""

TAGLINE = "Find the best Ollama model for your hardware"


def _print_header() -> None:
    """Print the app header."""
    logo_text = Text(LOGO, style="bold cyan")
    tagline_text = Text(f"  {TAGLINE}\n", style="dim white")

    console.print(logo_text, end="")
    console.print(tagline_text)


def _print_hardware_summary(hw: HardwareInfo) -> None:
    """Print a beautiful hardware summary panel."""
    items = []

    # OS
    os_display = f"{hw.os_name} {hw.os_version}" if hw.os_version else hw.os_name
    items.append(f"  🖥️  [bold]System[/]     {os_display}")

    # CPU
    items.append(f"  ⚡  [bold]CPU[/]        {hw.cpu_name}")
    items.append(f"                  {hw.cpu_cores} cores / {hw.cpu_threads} threads ({hw.cpu_arch})")

    # RAM
    ram_color = "green" if hw.ram_total_gb >= 32 else "yellow" if hw.ram_total_gb >= 16 else "red"
    items.append(
        f"  🧠  [bold]RAM[/]        [{ram_color}]{hw.ram_total_gb:.1f} GB[/{ram_color}] total"
        f" · {hw.ram_available_gb:.1f} GB available"
    )

    # GPU
    if hw.is_apple_silicon and hw.gpus:
        gpu = hw.gpus[0]
        items.append(
            f"  🎮  [bold]GPU[/]        {gpu.name} (Unified Memory)"
        )
        items.append(
            f"                  ~{hw.effective_vram_gb:.1f} GB available for models"
        )
    elif hw.gpus:
        for gpu in hw.gpus:
            vram_color = "green" if gpu.vram_gb >= 16 else "yellow" if gpu.vram_gb >= 8 else "red"
            items.append(
                f"  🎮  [bold]GPU[/]        {gpu.name}"
                f" · [{vram_color}]{gpu.vram_gb:.1f} GB VRAM[/{vram_color}]"
            )
    else:
        items.append("  🎮  [bold]GPU[/]        [dim]No dedicated GPU detected[/dim]")

    # Inference mode
    mode_color = "green" if hw.has_gpu else "yellow"
    items.append(f"  🔮  [bold]Inference[/]  [{mode_color}]{hw.inference_mode}[/{mode_color}]")

    # Disk
    items.append(f"  💿  [bold]Disk[/]       {hw.disk_available_gb:.1f} GB free")

    panel_content = "\n".join(items)
    console.print(Panel(
        panel_content,
        title="[bold white]📋 Your Hardware[/]",
        border_style="cyan",
        padding=(1, 2),
    ))


def _print_ollama_status() -> None:
    """Check and display Ollama installation status."""
    with console.status("[cyan]Checking Ollama installation...", spinner="dots"):
        installed = is_ollama_installed()
        time.sleep(0.3)

    if installed:
        version = get_ollama_version()
        ver_display = f" (v{version})" if version else ""
        console.print(f"  ✅ Ollama is installed{ver_display}\n", style="green")

        # Show installed models
        models = get_installed_models()
        if models:
            console.print(f"  📦 You already have [bold]{len(models)}[/] model(s) installed:")
            for m in models:
                size_info = f" ({m['size']})" if m.get('size') else ""
                console.print(f"     • {m['name']}{size_info}", style="dim")
            console.print()
    else:
        console.print(
            Panel(
                "  Ollama is not installed. Install it first:\n\n"
                "  [bold cyan]curl -fsSL https://ollama.com/install.sh | sh[/]\n\n"
                "  Or visit [link=https://ollama.com]https://ollama.com[/link]",
                title="[bold yellow]⚠️  Ollama Not Found[/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        console.print()


def _print_top_pick(rec: Recommendation) -> None:
    """Print the top recommendation prominently."""
    model = rec.model

    stars = "⭐" * model.quality_tier

    content_lines = [
        "",
        f"  [bold white]{model.full_name}[/] — {model.description}",
        "",
        f"  📊  Parameters:   {model.param_label}",
        f"  💾  Download:     {model.size_gb} GB",
        f"  🧠  Memory:      {model.vram_gb} GB needed",
        f"  {rec.performance_icon}  Performance:  [bold]{rec.performance.upper()}[/]",
        f"  🏆  Quality:      {stars}",
        f"  🏷️   Capabilities: {model.capability_icons}  ({', '.join(model.capabilities)})",
        "",
        f"  [bold green]Get started:[/]",
        f"  [bold cyan]  ollama pull {model.full_name}[/]",
        f"  [bold cyan]  ollama run {model.full_name}[/]",
        "",
    ]

    console.print(Panel(
        "\n".join(content_lines),
        title="[bold white]🏆 Top Recommendation[/]",
        border_style="green",
        padding=(0, 2),
    ))


def _print_all_compatible(recommendations: list[Recommendation]) -> None:
    """Print a table of all compatible models."""
    if not recommendations:
        console.print(Panel(
            "  No compatible models found for your hardware.\n"
            "  Try freeing up memory or upgrading your RAM.",
            title="[bold red]😔 No Models Found[/]",
            border_style="red",
        ))
        return

    table = Table(
        title="All Compatible Models",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        show_lines=True,
        padding=(0, 1),
    )

    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Model", style="bold white", min_width=20)
    table.add_column("Params", justify="right", min_width=7)
    table.add_column("Size", justify="right", min_width=7)
    table.add_column("Memory", justify="right", min_width=8)
    table.add_column("Speed", justify="center", min_width=8)
    table.add_column("Quality", justify="center", min_width=9)
    table.add_column("Type", min_width=12)

    for i, rec in enumerate(recommendations, 1):
        model = rec.model
        stars = "⭐" * model.quality_tier

        # Speed display
        speed_display = f"{rec.performance_icon} {rec.performance}"

        table.add_row(
            str(i),
            f"{model.full_name}\n[dim]{model.description}[/]",
            model.param_label,
            f"{model.size_gb} GB",
            f"{model.vram_gb} GB",
            speed_display,
            stars,
            model.capability_icons,
        )

    console.print(table)


def _print_category_picks(recommendations: list[Recommendation]) -> None:
    """Print the best model for each category."""
    categories = get_recommendations_by_category(recommendations)

    # Category display config
    category_info = {
        "chat": ("💬 Best for Chat", "General conversation, Q&A, writing"),
        "code": ("💻 Best for Code", "Programming, debugging, code review"),
        "vision": ("👁️  Best for Vision", "Image understanding, visual Q&A"),
        "reasoning": ("🧠 Best for Reasoning", "Complex logic, math, analysis"),
        "rag": ("📚 Best for RAG", "Document Q&A, knowledge retrieval"),
        "math": ("🔢 Best for Math", "Mathematical problem solving"),
    }

    panels = []
    for cap, (title, desc) in category_info.items():
        if cap not in categories:
            continue

        best = categories[cap][0]  # Already sorted by quality
        model = best.model

        panel_content = (
            f"[bold]{model.full_name}[/]\n"
            f"{model.param_label} · {model.size_gb} GB · {best.performance_icon}\n"
            f"[dim]{desc}[/]"
        )

        panels.append(Panel(
            panel_content,
            title=f"[bold]{title}[/]",
            border_style="blue",
            width=36,
            padding=(1, 2),
        ))

    if panels:
        console.print()
        console.print("[bold white]  📂 Best Pick by Category[/]\n")
        # Print panels in rows of 3
        for i in range(0, len(panels), 3):
            row = panels[i:i+3]
            console.print(Columns(row, padding=(0, 1), expand=False))


def _print_quick_start_guide() -> None:
    """Print helpful quick start instructions."""
    guide = (
        "  [bold white]🚀 Quick Start Guide[/]\n"
        "\n"
        "  [bold]1.[/] Install a model:    [cyan]ollama pull <model>[/]\n"
        "  [bold]2.[/] Chat with it:       [cyan]ollama run <model>[/]\n"
        "  [bold]3.[/] List your models:   [cyan]ollama list[/]\n"
        "  [bold]4.[/] Stop a model:       [cyan]ollama stop <model>[/]\n"
        "\n"
        "  [dim]💡 Tip: Start with the Top Recommendation above — it's the\n"
        "     best balance of quality and performance for your hardware![/]"
    )

    console.print(Panel(
        guide,
        border_style="dim",
        padding=(1, 2),
    ))


def _print_tips(hw: HardwareInfo) -> None:
    """Print hardware-specific tips."""
    tips = []

    if not hw.has_gpu:
        tips.append(
            "🔧 [bold]No GPU detected[/] — Models will run on CPU, which is slower. "
            "Consider a GPU with at least 8 GB VRAM for a much better experience."
        )

    if hw.ram_total_gb < 16:
        tips.append(
            "💡 [bold]Low RAM[/] — You have less than 16 GB RAM. "
            "Stick to smaller models (≤ 3B parameters) for the best experience."
        )

    if hw.is_apple_silicon:
        tips.append(
            "🍎 [bold]Apple Silicon[/] — Great news! Ollama uses Metal acceleration, "
            "so your unified memory works as both RAM and VRAM. You'll get excellent performance."
        )

    if hw.effective_vram_gb >= 24:
        tips.append(
            "🚀 [bold]High-end setup[/] — You have enough memory for larger models (27B+). "
            "These offer significantly better quality for complex tasks."
        )

    if hw.disk_available_gb < 20:
        tips.append(
            "⚠️  [bold]Low disk space[/] — Models can be large (up to 43 GB for 70B). "
            "Consider freeing up some space if you want to try bigger models."
        )

    if tips:
        console.print()
        for tip in tips:
            console.print(Padding(f"  {tip}", (0, 2)))
        console.print()


def _print_mode_selection() -> str:
    """
    Ask the user to choose between recommendations or model search.

    Returns "recommend" or "search".
    """
    console.print()
    console.print(Panel(
        "  [bold]What would you like to do?[/]\n\n"
        "  [bold cyan][1][/] [white]Get recommendations[/] \u2014 see the best models for your hardware\n"
        "  [bold cyan][2][/] [white]Search a model[/]    \u2014 check if a specific model can run on your system",
        title="[bold white]Choose a Mode[/]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()

    while True:
        choice = console.input("  [bold cyan]Enter 1 or 2:[/] ").strip()
        if choice in ("1", ""):
            console.print()
            return "recommend"
        if choice == "2":
            console.print()
            return "search"
        console.print("  [yellow]Please enter 1 or 2.[/]")


def _run_search_mode(hw: HardwareInfo, all_models: list[OllamaModel]) -> None:
    """Interactive model search: user types a name, we show compatibility."""
    effective_vram = hw.effective_vram_gb

    console.print("  [dim]Tip: search by partial name, e.g. 'qwen', 'llama', 'gemma', 'deepseek'[/]\n")

    while True:
        query = console.input(
            "  [bold cyan]Search model name (or [bold]q[/] to quit):[/] "
        ).strip().lower()

        if query in ("q", "quit", "exit", ""):
            break

        # Find matching models (name contains the query)
        matches = [m for m in all_models if query in m.name.lower() or query in m.full_name.lower()]

        if not matches:
            console.print(f"\n  [yellow]No models found matching '[bold]{query}[/]'.[/]")
            console.print("  [dim]Tip: Try a partial name, e.g. 'qwen', 'llama', 'gemma'[/]\n")
            continue

        console.print()

        # Group by model name
        names: dict[str, list[OllamaModel]] = {}
        for m in matches:
            names.setdefault(m.name, []).append(m)

        for name, variants in names.items():
            # Use first variant's description (strip the parenthesised size if present)
            raw_desc = variants[0].description
            short_desc = raw_desc.split(" (")[0] if " (" in raw_desc else raw_desc[:60]

            table = Table(
                title=f"{name} \u2014 {short_desc}",
                box=box.ROUNDED,
                header_style="bold cyan",
                border_style="dim",
                show_lines=True,
                padding=(0, 1),
            )
            table.add_column("Variant", style="bold white", min_width=16)
            table.add_column("Params", justify="right", min_width=7)
            table.add_column("Download", justify="right", min_width=9)
            table.add_column("VRAM needed", justify="right", min_width=11)
            table.add_column("Can run?", justify="center", min_width=12)
            table.add_column("Speed", justify="center", min_width=10)
            table.add_column("Type", min_width=8)

            for m in sorted(variants, key=lambda x: x.parameters):
                fits_vram = m.vram_gb <= effective_vram
                fits_disk = m.size_gb <= hw.disk_available_gb

                if not fits_vram:
                    shortage = m.vram_gb - effective_vram
                    can_run = "[red]\u2717 Too large[/]"
                    speed_label = "\u2014"
                    vram_str = f"[red]{m.vram_gb} GB (+{shortage:.1f})[/]"
                elif not fits_disk:
                    can_run = "[yellow]\u2717 No disk[/]"
                    speed_label = "\u2014"
                    vram_str = f"{m.vram_gb} GB"
                else:
                    perf, icon, _ = _classify_performance(m, effective_vram, hw.has_gpu)
                    can_run = "[green]\u2713 Yes[/]"
                    speed_label = f"{icon} {perf}"
                    vram_str = f"{m.vram_gb} GB"

                table.add_row(
                    m.full_name,
                    m.param_label,
                    f"{m.size_gb} GB",
                    vram_str,
                    can_run,
                    speed_label,
                    m.capability_icons,
                )

            console.print(table)
            console.print()


def main() -> None:
    """Main entry point for the CLI."""
    console.print()
    _print_header()

    # ── Step 1: Detect hardware ──────────────────────────────────────
    with console.status(
        "[bold cyan]  🔍 Scanning your hardware...",
        spinner="dots",
    ):
        time.sleep(0.5)  # Brief pause for visual feedback
        hw = detect_hardware()

    _print_hardware_summary(hw)
    console.print()

    # ── Step 2: Check Ollama ─────────────────────────────────────────
    _print_ollama_status()

    # ── Step 3: Mode selection ───────────────────────────────────────
    mode = _print_mode_selection()

    # ── Step 4: Fetch all models ─────────────────────────────────────
    with console.status(
        "[bold cyan]  🌐 Fetching latest models from the Ollama library...",
        spinner="dots",
    ):
        all_models = get_all_models()

    if not all_models:
        console.print(Panel(
            "\n  ⚠️  Could not fetch models from the Ollama library.\n\n"
            "  This usually means you are offline or ollama.com is unreachable.\n"
            "  Please check your internet connection and try again.\n",
            title="[bold yellow]No Models Available[/]",
            border_style="yellow",
            padding=(1, 2),
        ))
        console.print()
        return

    # ── Step 5: Run selected mode ────────────────────────────────────
    if mode == "search":
        _run_search_mode(hw, all_models)
    else:
        # Recommendations mode
        from localai.recommender import get_recommendations, get_top_pick
        recommendations = get_recommendations(hw)
        top = get_top_pick(recommendations)

        if top:
            _print_top_pick(top)
            console.print()
            _print_category_picks(recommendations)
            console.print()
            _print_all_compatible(recommendations)
            console.print()
            _print_quick_start_guide()
        else:
            console.print(Panel(
                "\n  😔 Unfortunately, no Ollama models seem to be compatible\n"
                "  with your current hardware.\n\n"
                "  [bold]Possible solutions:[/]\n"
                "  • Close other applications to free up RAM\n"
                "  • Upgrade to at least 8 GB RAM\n"
                "  • Add a GPU with at least 4 GB VRAM\n",
                title="[bold red]No Compatible Models[/]",
                border_style="red",
                padding=(1, 2),
            ))

        # ── Hardware-specific tips ───────────────────────────────────
        _print_tips(hw)

        # ── Footer ──────────────────────────────────────────────────
        console.print(
            "  [dim]Made with ❤️  by LocalAI · "
            f"{len(recommendations)} models compatible with your hardware[/]",
        )
        console.print()


if __name__ == "__main__":
    main()
