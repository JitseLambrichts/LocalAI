#!/usr/bin/env bash

# Navigeer naar de map van het script
cd "$(dirname "$0")"

# Activeer de virtual environment
source venv/bin/activate

# Voer de applicatie uit
python3 -m localai
