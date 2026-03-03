"""
Ollama detection utilities.

Checks if Ollama is installed and which models are already downloaded.
"""

import json

from localai.hardware import _run_command


def is_ollama_installed() -> bool:
    """Check if Ollama is installed and accessible."""
    output = _run_command(["ollama", "--version"])
    return bool(output)


def get_ollama_version() -> str:
    """Get the installed Ollama version string."""
    output = _run_command(["ollama", "--version"])
    if output:
        # Output is usually like "ollama version is 0.1.45"
        return output.replace("ollama version is ", "").strip()
    return ""


def get_installed_models() -> list[dict]:
    """
    Get list of models already installed locally.

    Returns a list of dicts with keys: name, size, modified.
    """
    output = _run_command(["ollama", "list"])
    if not output:
        return []

    models = []
    lines = output.strip().splitlines()

    # Skip header row
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0]
            # Try to find size (look for pattern like "4.9 GB")
            size_str = ""
            for i, part in enumerate(parts):
                if part in ("GB", "MB", "KB") and i > 0:
                    size_str = f"{parts[i-1]} {part}"
                    break

            models.append({
                "name": name,
                "size": size_str,
            })

    return models


def get_running_models() -> list[str]:
    """Get list of currently running/loaded models."""
    output = _run_command(["ollama", "ps"])
    if not output:
        return []

    models = []
    lines = output.strip().splitlines()

    for line in lines[1:]:  # Skip header
        parts = line.split()
        if parts:
            models.append(parts[0])

    return models
