# LoRA Vision Captioner

**Quick Install (Recommended):**

```bash
git clone <your-repo-url> lora-captioner
cd lora-captioner
./install.sh
```

### API Key Setup (for Grok)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your xAI API key:
   ```env
   XAI_API_KEY=your_key_here
   ```

Get your free API key at: [https://console.x.ai/](https://console.x.ai/)

---

A powerful interactive tool for generating high-quality image descriptions/captions for **LoRA training** using vision-language models.

Supports three backends:
- **Grok** (xAI) – via official API
- **Ollama** – local models
- **vLLM** – local OpenAI-compatible server (great for running large VLMs)

## Key Features

- **Interactive Model Selection Wizard**: Automatically queries your Ollama or vLLM server and lets you pick from available models.
- **Vision Model Detection**: Smartly identifies and prioritizes vision-capable models (shows `[VISION]` tag). Defaults to showing only vision models for image captioning use cases.
- **Multi-backend support**: Easily switch between Grok, Ollama, and vLLM.
- **Smart WSL / Host IP handling** for vLLM (auto-detects Windows host IP when running in WSL).
- **LoRA-optimized prompting**: Built-in system prompt tuned for generating rich training captions.
- **Resume-safe + Dry-run support**.

## Installation

The easiest way is to use the provided installer:

```bash
./install.sh
```

This script will:
- Install Python 3 (via apt) if it is missing
- Create a virtual environment (`.venv`)
- Install all required dependencies

After running the installer, activate the environment:

```bash
source .venv/bin/activate
python lora_captioner.py --help
```

### Setting up your Grok (xAI) API Key

The script supports multiple ways to provide the API key:

1. **Recommended**: Create a `.env` file in the project root
   ```bash
   cp .env.example .env
   # Then edit .env and add your key
   ```

2. Environment variable:
   ```bash
   export XAI_API_KEY=your_key_here
   ```

3. Command line flag:
   ```bash
   python lora_captioner.py --backend grok --api-key your_key_here ...
   ```

> **Note**: The `.env` file is ignored by git (see `.gitignore`). Never commit your real API key.

## Usage

### Basic usage (with interactive wizard)

```bash
# Ollama
python lora_captioner.py --backend ollama --folder /path/to/images --skip-qa

# vLLM
python lora_captioner.py --backend vllm --folder /path/to/images --skip-qa

# Grok (xAI)
python lora_captioner.py --backend grok --folder /path/to/images --skip-qa
```

The wizard will automatically appear for Ollama and vLLM if you don't specify `--model`.

### Skip the wizard

```bash
python lora_captioner.py --backend ollama --model llava:13b --folder ./images
```

### Useful flags

| Flag              | Description                                      |
|-------------------|--------------------------------------------------|
| `--backend`       | `grok`, `ollama`, or `vllm`                      |
| `--model`         | Skip wizard and use specific model               |
| `--vllm-ip`       | Manually specify IP when using vLLM from WSL     |
| `--skip-qa`       | Skip files with `__qa` in the name               |
| `--glob`          | Filter images (e.g. `' *__R.png'`)               |
| `--dry-run`       | Preview without generating captions              |
| `--system-add`    | Append extra instructions to the system prompt   |
| `--max-side`      | Resize images before sending (saves tokens)      |

## Uninstallation

To remove the virtual environment:

```bash
./uninstall.sh
```

## Recommended Workflow for LoRA Training

1. Generate captions using a strong vision model (e.g. `llava`, `qwen2.5-vl`, or Grok).
2. Use `--glob '*__R.png'` or similar to only caption your best renders.
3. Use `--system-add` to inject character-specific details.
4. Review generated `.txt` files before training.

## Why This Tool?

Most generic image captioning tools are not optimized for **character LoRA training**. This tool was built specifically to produce rich, consistent, subject-focused descriptions that help diffusion models learn specific characters well.

## License

MIT
