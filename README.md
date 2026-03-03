# LocalAI 💻🤖

LocalAI is a powerful Command Line Interface (CLI) application that assesses your computer's hardware and recommends the most suitable AI models for local execution through [Ollama](https://ollama.ai/). It analyzes your CPU, GPU, and RAM to determine whether your system can optimally run smaller, medium, or large AI models locally without performance bottlenecks.

## 🚀 Getting Started

Follow these steps to run LocalAI on your machine.

### Prerequisites

- Python 3.9 or higher
- Git

### Installation & Execution

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JitseLambrichts/LocalAI.git
   cd LocalAI
   ```

2. **Run the application (macOS / Linux):**
   We have provided a convenient shell script that sets up the environment and executes the app.

   If it's the first time you are running it, give the script execute permissions:

   ```bash
   chmod +x run.sh
   ```

   Then, execute the script:

   ```bash
   ./run.sh
   ```

   _Alternatively_, if you want to set it up manually:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -e .
   python3 -m localai
   ```

## 🛠️ What it does

Running Large Language Models (LLMs) locally can be demanding on your computer's resources. LocalAI simplifies this by doing the heavy lifting for you:

1. **Hardware Detection**: It intelligently scans your specific hardware setup, accurately identifying your CPU architecture, the amount of available System RAM (Memory), and whether you have a GPU (like Apple Silicon Unified Memory or dedicated graphics).
2. **Performance Assessment**: Based on the technical specs of your system, LocalAI evaluates the capabilities. Model sizes like 7B, 13B, or 70B parameters have vastly different hardware requirements.
3. **Model Recommendation**: It provides you with a rich terminal interface that neatly presents safe and optimal Ollama models to run on your device. Whether you can handle heavy-weight models or need high-efficiency smaller models, LocalAI will present the perfect options.

## 🧰 Features under the hood

- Built with `psutil` for highly accurate system metrics and hardware checking across platforms.
- Employs `rich` to provide a visually appealing, colorful, and highly readable console output.
- A highly modular and clean codebase (`hardware.py`, `models.py`, `recommender.py`) ensuring easy maintainability and extensibility.

## ⚖️ License

LocalAI is open source and distributed under the MIT License. Feel free to fork, expand, and contribute!
