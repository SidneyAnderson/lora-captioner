#!/usr/bin/env python3
"""
lora_captioner.py

Multi-backend vision captioner for LoRA training data.

Supports:
  - Grok (xAI) via official API
  - Local Ollama
  - vLLM (OpenAI-compatible server, e.g. Qwen2-VL captioning models)

Features:
  - Generalized high-quality LoRA caption prompt (not tied to any specific character)
  - --system-add: Append custom instructions to the base system prompt
  - Interactive model selection wizard for Ollama and vLLM (queries the server and lets you pick)
  - .env file support for keys
  - Image resizing before upload (saves tokens/cost with Grok)
  - Resume-safe, dry-run, glob filtering, etc.

Usage examples:

    # Using Grok (default)
    python lora_captioner.py --folder . --skip-qa --system-add "The main subject is Red_Fairy_HH, a red-haired fairy with large iridescent wings."

    # Using local Ollama
    python lora_captioner.py --backend ollama --skip-qa

    # Using vLLM (most common when vLLM runs inside WSL)
    # → Interactive wizard will show available models and let you choose
    python lora_captioner.py --backend vllm --skip-qa

    # When vLLM runs on the Windows host instead (from WSL)
    python lora_captioner.py --backend vllm --vllm-ip 172.20.10.5 --skip-qa

    # Skip the wizard by specifying the model directly
    python lora_captioner.py --backend vllm --model your-loaded-model --skip-qa

    # With extra prompt tuning
    python lora_captioner.py --backend vllm --system-add "Focus on detailed clothing and expression."

Requirements:
    pip install openai pillow requests

For vLLM backend you also need a running vLLM server:
    vllm serve prithivMLmods/Qwen2-VL-2B-Abliterated-Caption-it --host 0.0.0.0 --port 8000

Common setups:
  - Run vLLM inside WSL → script auto-detects IP using 'hostname -I'
  - Run vLLM on Windows host → use --vllm-ip <your-windows-ip> from WSL
"""

import argparse
import base64
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Tuple

# =============================================================================
# DEPENDENCIES
# =============================================================================

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required. Install with: pip install requests")
    sys.exit(1)

try:
    from openai import OpenAI
    import httpx
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# .env LOADER (same as before)
# =============================================================================

def load_env_file(env_path: Path) -> bool:
    if not env_path.exists():
        return False
    loaded = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded = True
    return loaded


def _translate_windows_path(path_arg: str) -> Optional[Path]:
    r"""
    Translate common Windows path forms into paths usable from WSL/Linux.

    Handles:
    - \\wsl.localhost\Distro\home\user\images -> /home/user/images
    - //wsl.localhost/Distro/home/user/images -> /home/user/images
    - C:\Users\name\images -> /mnt/c/Users/name/images
    """
    normalized = path_arg.replace("\\", "/")
    lowered = normalized.lower()

    for prefix in ("//wsl.localhost/", "//wsl$/", "/wsl.localhost/", "/wsl$/"):
        if lowered.startswith(prefix):
            rest = normalized[len(prefix):]
            parts = [part for part in rest.split("/") if part]
            if len(parts) >= 2:
                return Path("/" + "/".join(parts[1:]))

    if (
        len(path_arg) >= 3
        and path_arg[0].isalpha()
        and path_arg[1] == ":"
        and path_arg[2] in ("\\", "/")
    ):
        drive = path_arg[0].lower()
        rest = path_arg[2:].replace("\\", "/").lstrip("/")
        return Path("/mnt") / drive / rest

    return None


def _recover_existing_path_from_compact(root: Path, compact_path: str, depth: int = 0) -> Optional[Path]:
    if not compact_path:
        return root if root.is_dir() else None
    if depth > 64 or not root.is_dir():
        return None

    try:
        entries = sorted(root.iterdir(), key=lambda p: len(p.name), reverse=True)
    except OSError:
        return None

    compact_lower = compact_path.lower()
    for entry in entries:
        if not entry.is_dir():
            continue
        if compact_lower.startswith(entry.name.lower()):
            recovered = _recover_existing_path_from_compact(
                entry,
                compact_path[len(entry.name):],
                depth + 1,
            )
            if recovered:
                return recovered

    return None


def _recover_shell_stripped_wsl_unc_path(path_arg: str) -> Optional[Path]:
    r"""
    Bash removes unquoted backslashes, so a command like:
      -f \\wsl.localhost\Ubuntu-22.04\home\sid\images

    reaches Python as one compact string with most separators missing. If the
    target directory already exists, recover it by matching path components.
    """
    compact = path_arg.replace("\\", "").replace("/", "")
    compact_lower = compact.lower()

    for host_prefix in ("wsl.localhost", "wsl$"):
        if not compact_lower.startswith(host_prefix):
            continue

        rest = compact[len(host_prefix):]
        distro = os.environ.get("WSL_DISTRO_NAME")
        if distro and rest.lower().startswith(distro.lower()):
            rest = rest[len(distro):]

        # If the distro name was not available or did not match, scan for the
        # first suffix that maps to a real absolute path such as /home/... .
        for index in range(len(rest)):
            recovered = _recover_existing_path_from_compact(Path("/"), rest[index:])
            if recovered:
                return recovered

    return None


def resolve_cli_path(path_arg: str) -> Path:
    translated = _translate_windows_path(path_arg)
    if translated:
        return translated.expanduser().resolve()

    recovered = _recover_shell_stripped_wsl_unc_path(path_arg)
    if recovered:
        return recovered.expanduser().resolve()

    return Path(path_arg).expanduser().resolve()


def print_folder_error(raw_folder: str, folder: Path) -> None:
    print("ERROR: Image folder does not exist.")
    print(f"  Input:    {raw_folder}")
    print(f"  Resolved: {folder}")

    compact = raw_folder.replace("\\", "").replace("/", "").lower()
    if compact.startswith(("wsl.localhost", "wsl$")):
        home = Path.home()
        distro = os.environ.get("WSL_DISTRO_NAME", "Ubuntu-22.04")
        print("")
        print("This looks like a Windows WSL UNC path that Bash may have stripped.")
        print("From WSL, prefer the native Linux path:")
        print(f"  python3 lora_captioner.py -f {home}/path/to/images")
        print("")
        print("Or quote the UNC path so Bash preserves the backslashes:")
        print(f"  python3 lora_captioner.py -f '\\\\wsl.localhost\\{distro}\\home\\{home.name}\\path\\to\\images'")


def get_vllm_base_url(vllm_ip: Optional[str] = None, default_url: str = "http://localhost:8000/v1") -> str:
    """
    Resolve the correct base URL for vLLM.

    Two common scenarios:
    - vLLM running inside WSL (same machine as the script): uses auto-detection (hostname -I first).
    - vLLM running on Windows host: user should pass --vllm-ip <windows-ip>.

    Priority:
    1. If --vllm-ip is provided → use it
    2. If inside WSL → auto-detect using hostname -I, then ip route, then resolv.conf
    3. Fall back to --vllm-url
    """
    if vllm_ip:
        return f"http://{vllm_ip}:8000/v1"

    # Try to detect if we are inside WSL
    try:
        if os.path.exists("/proc/version"):
            with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
                if "microsoft" in f.read().lower():
                    # Method 1: hostname -I (as requested)
                    try:
                        result = subprocess.run(
                            ["hostname", "-I"],
                            capture_output=True, text=True, timeout=1.5
                        )
                        if result.returncode == 0:
                            ips = result.stdout.strip().split()
                            if ips:
                                host_ip = ips[0]
                                print(f"Auto-detected IP via hostname -I: {host_ip}")
                                return f"http://{host_ip}:8000/v1"
                    except Exception:
                        pass

                    # Method 2: ip route (default gateway)
                    try:
                        result = subprocess.run(
                            ["ip", "route", "show", "default"],
                            capture_output=True, text=True, timeout=1.5
                        )
                        if result.returncode == 0:
                            parts = result.stdout.strip().split()
                            if len(parts) >= 3:
                                host_ip = parts[2]
                                print(f"Auto-detected Windows host IP via ip route: {host_ip}")
                                return f"http://{host_ip}:8000/v1"
                    except Exception:
                        pass

                    # Method 3: /etc/resolv.conf (nameserver)
                    try:
                        with open("/etc/resolv.conf", "r", encoding="utf-8", errors="ignore") as resolv:
                            for line in resolv:
                                if "nameserver" in line:
                                    ip = line.split()[1].strip()
                                    if ip and not ip.startswith("127."):
                                        print(f"Auto-detected Windows host IP via resolv.conf: {ip}")
                                        return f"http://{ip}:8000/v1"
                    except Exception:
                        pass
    except Exception:
        pass

    return default_url


# =============================================================================
# MODEL DISCOVERY WIZARD (for Ollama and vLLM)
# =============================================================================

# Known vision / multimodal model name patterns (case-insensitive)
# These are models that can accept images as input.
VISION_MODEL_PATTERNS = [
    "llava",
    "vision",
    "vl",
    "moondream",
    "bakllava",
    "minicpm-v",
    "qwen2-vl",
    "qwen2.5-vl",
    "phi-3-vision",
    "phi3-vision",
    "granite3.2-vision",
    "internvl",
    "idefics",
    "cogvlm",
    "llava-llama3",
    "llava-phi3",
]


def is_vision_model(name: str, details: dict = None) -> bool:
    """
    Heuristic to determine if an Ollama model is likely vision-capable
    (can accept images).
    """
    name_lower = name.lower()

    for pattern in VISION_MODEL_PATTERNS:
        if pattern in name_lower:
            return True

    # Also check family from details if available
    if details:
        family = (details.get("family") or "").lower()
        if family in ["llava", "qwen2", "moondream"]:
            return True

    return False


def format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable size string."""
    if size_bytes is None or size_bytes == 0:
        return "?"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def list_ollama_models(host: str) -> list[dict]:
    """
    Query Ollama and return rich model info.
    Each entry contains: name, size, size_str, details_str, display
    """
    url = f"{host.rstrip('/')}/api/tags"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size = m.get("size", 0)
            details = m.get("details", {}) or {}

            parameter_size = details.get("parameter_size", "")
            quantization = details.get("quantization_level", "")
            family = details.get("family", "")

            size_str = format_size(size)

            extra = []
            if parameter_size:
                extra.append(parameter_size)
            if quantization:
                extra.append(quantization)
            if family:
                extra.append(family)

            details_str = ", ".join(extra) if extra else ""

            vision_capable = is_vision_model(name, details)

            display = f"{name}"
            if details_str:
                display += f"  ({details_str})"
            display += f"  [{size_str}]"

            if vision_capable:
                display += "   [VISION]"

            models.append({
                "name": name,
                "size": size,
                "size_str": size_str,
                "details_str": details_str,
                "is_vision": vision_capable,
                "display": display
            })

        return models

    except Exception as e:
        print(f"Warning: Could not query Ollama at {host}: {e}")
        return []


def list_vllm_models(base_url: str) -> list[dict]:
    """
    Query vLLM and return model info.
    vLLM returns limited info, so we mostly just have the model ID.
    """
    if not HAS_OPENAI:
        print("Warning: openai package not available, cannot list vLLM models.")
        return []
    try:
        client = OpenAI(api_key="EMPTY", base_url=base_url, timeout=10)
        models = client.models.list()

        result = []
        for m in models.data:
            # Most people use vLLM to serve vision models, so we mark them as vision-capable
            result.append({
                "name": m.id,
                "size": 0,
                "size_str": "?",
                "details_str": "",
                "is_vision": True,
                "display": f"{m.id}   [VISION]"
            })
        return result
    except Exception as e:
        print(f"Warning: Could not query vLLM at {base_url}: {e}")
        return []


def select_model_wizard(backend: str, ollama_host: str, vllm_base_url: Optional[str]) -> str:
    """
    Interactive wizard to let the user choose a model when none was specified.
    Shows extra info (size, parameters, quantization) when available (especially for Ollama).
    """
    print("\n" + "=" * 60)
    print(f"  {backend.upper()} MODEL SELECTION WIZARD")
    print("=" * 60)

    if backend == "ollama":
        print(f"Querying Ollama at: {ollama_host}")
        models = list_ollama_models(ollama_host)
        server_name = "Ollama"
    else:  # vllm
        print(f"Querying vLLM at: {vllm_base_url}")
        models = list_vllm_models(vllm_base_url) if vllm_base_url else []
        server_name = "vLLM"

    if not models:
        print(f"\nNo models found on the {server_name} server.")
        print("Please make sure the server is running and has models loaded.")
        model = input(f"\nEnter the exact model name to use for {backend}: ").strip()
        return model

    # Separate vision and non-vision models
    vision_models = [m for m in models if m.get("is_vision")]
    text_models = [m for m in models if not m.get("is_vision")]

    show_vision_only = True
    if text_models:
        # For image-to-text / LoRA captioning use case, default to vision-only
        print(f"\nDetected {len(vision_models)} vision-capable model(s) and {len(text_models)} text-only model(s).")
        choice = input("Show only vision models? [Y/n]: ").strip().lower()
        if choice == "n":
            show_vision_only = False

    display_list = vision_models if show_vision_only else models

    if not display_list:
        print("\nNo vision models found. Showing all models instead.")
        display_list = models

    print(f"\nAvailable models on {server_name}:\n")
    for i, m in enumerate(display_list, 1):
        tag = ""
        if m.get("is_vision"):
            tag = "  [VISION]"
        print(f"  [{i}] {m['display']}{tag}")

    print("\n  [0] Enter a custom model name manually")

    max_choice = len(display_list)

    while True:
        prompt = f"\nSelect a model [1-{max_choice}] or 0 for custom: "
        choice = input(prompt).strip()

        if choice == "0":
            custom = input("Enter custom model name: ").strip()
            if custom:
                return custom
            print("Invalid input.")
            continue

        try:
            idx = int(choice)
            if 1 <= idx <= max_choice:
                selected = display_list[idx - 1]
                if not selected.get("is_vision") and backend in ("ollama", "vllm"):
                    confirm = input(
                        f"\nWarning: '{selected['name']}' does not look like a vision model.\n"
                        "It may not support image inputs. Continue anyway? [y/N]: "
                    ).strip().lower()
                    if confirm != "y":
                        continue
                return selected["name"]
            else:
                print(f"Please enter a number between 1 and {max_choice}, or 0.")
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# GENERALIZED BASE PROMPT (not character-specific)
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an expert visual descriptor specializing in high-quality training captions for LoRA and DreamBooth fine-tuning of diffusion models (Z-Image Turbo, Flux, SDXL, Pony, Aurora, etc.).

Your job is to produce a single, vivid, natural-language caption that teaches the model the exact visual identity of the subject and not the specific scene. 
Core rules:
- Write in flowing, prosaic English (1–3 paragraphs max). Never use bullet points or tag lists.
- Never start with "This image shows...", "A photo of...", "The image depicts...", or similar meta phrases.
- Be extremely specific and vivid about the character: hair (color, style, length, volume, texture), face (eye color, expression, makeup, facial structure), body (proportions, skin, pose, hands, feet), distinctive traits (fairy wings, ears, etc.).
- Describe clothing state accurately (nude, partially clothed, specific outfit details).
- Describe nude anatomy with care and precision, including any visible genitalia, breasts, nipples, pubic hair, etc. Use anatomically, slang and common terms.
- Any time 'vulva' is used replace with 'pussy' and describe in detail (labia, clitoris, etc.) unless the image is explicitly clinical or artistic in nature. If the image is a clinical or artistic nude study, 'vulva' is acceptable.
- Any time 'penis' is used replace with 'cock' and describe in detail (balls, shaft, head, etc.) unless the image is explicitly clinical or artistic in nature. If the image is a clinical or artistic nude study, 'penis' is acceptable.
- Describe the environment, lighting, mood, camera angle/framing/composition, depth of field, and artistic/rendering style.
- Focus on what is visually distinctive and important for the model to learn.
- Keep the caption between 60–250 words. Quality and specificity matter more than length.
- Don't guess, only state what is factually understood.
- Don't include scene details
- Don't include pose details
- Dont include composition, such as lighting and shadow details
- Don't include background details

Output ONLY the caption text. Nothing else.

Provide a rich, detailed caption for this image suitable for LoRA training.

Output ONLY the caption text. Nothing else."""

DEFAULT_USER_PROMPT = "Provide a rich, detailed caption for this image suitable for LoRA training."


# =============================================================================
# IMAGE ENCODING
# =============================================================================

def encode_image(
    image_path: Path,
    max_side: Optional[int] = None,
    quality: int = 92
) -> Tuple[str, str]:
    """
    Returns (data_uri_for_grok, raw_base64_for_ollama)
    """
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"

    if max_side and HAS_PIL:
        with Image.open(image_path) as im:
            if suffix != ".png" and im.mode in ("RGBA", "P"):
                im = im.convert("RGB")

            if max(im.size) > max_side:
                ratio = max_side / max(im.size)
                new_size = (int(im.width * ratio), int(im.height * ratio))
                im = im.resize(new_size, Image.Resampling.LANCZOS)

            import io
            buffer = io.BytesIO()
            if suffix == ".png":
                im.save(buffer, format="PNG", optimize=True)
            else:
                im.save(buffer, format="JPEG", quality=quality, optimize=True)
            raw_bytes = buffer.getvalue()
    else:
        with open(image_path, "rb") as f:
            raw_bytes = f.read()

    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"
    return data_uri, b64


# =============================================================================
# BACKEND ABSTRACTION
# =============================================================================

def call_openai_compatible(
    client: OpenAI,
    model: str,
    system_prompt: str,
    image_path: Path,
    max_side: Optional[int],
    temperature: float,
) -> str:
    """Call any OpenAI-compatible vision endpoint (Grok, vLLM, etc.)."""
    data_uri, _ = encode_image(image_path, max_side=max_side)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": DEFAULT_USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ],
            },
        ],
        max_tokens=800,
        temperature=temperature,
        top_p=0.9,
    )
    return response.choices[0].message.content.strip()


def call_ollama(
    host: str,
    model: str,
    system_prompt: str,
    image_path: Path,
    max_side: Optional[int],
    temperature: float,
) -> str:
    """Call local Ollama vision model."""
    _, raw_b64 = encode_image(image_path, max_side=max_side)

    url = f"{host.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": DEFAULT_USER_PROMPT,
                "images": [raw_b64],
            },
        ],
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_predict": 800,
        },
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()

    # Ollama chat response structure
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"].strip()
    elif "response" in data:
        return data["response"].strip()
    else:
        raise RuntimeError(f"Unexpected Ollama response format: {data}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate high-quality LoRA training captions using Grok, Ollama, or vLLM vision models."
    )

    # Backend selection
    parser.add_argument(
        "--backend", choices=["grok", "ollama", "vllm"], default="grok",
        help="Which vision backend to use (default: grok)"
    )
    parser.add_argument(
        "-m", "--model",
        help="Model name. If omitted for ollama or vllm, an interactive wizard will query the server and let you choose."
    )

    # Ollama specific
    parser.add_argument(
        "--ollama-host", default="http://localhost:11434",
        help="Ollama server address (default: http://localhost:11434)"
    )

    # vLLM specific
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible server base URL (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--vllm-ip",
        help="IP address of the vLLM server. "
             "Required when vLLM runs on the Windows host while the script runs in WSL. "
             "When vLLM also runs inside WSL, the script auto-detects the IP using 'hostname -I' (first address)."
    )

    # Prompt control
    parser.add_argument(
        "--system-add", "--extra-prompt", dest="system_add",
        help="Additional text to APPEND to the base system prompt (ideal for character-specific details)"
    )
    parser.add_argument(
        "--system-prompt-file",
        help="Path to a text file containing a full system prompt (overrides the built-in one + --system-add)"
    )

    # General options (kept compatible with previous script)
    parser.add_argument("-f", "--folder", default=".", help="Folder containing images")
    parser.add_argument("--api-key", help="xAI API key (recommended: put in .env file as XAI_API_KEY)")
    parser.add_argument("--max-side", type=int, default=1280,
                        help="Resize longest side before sending to the vision model (0 = no resize)")
    parser.add_argument("--overwrite", "-o", action="store_true",
                        help="Overwrite existing .txt files")
    parser.add_argument("--skip-qa", action="store_true",
                        help="Skip files containing '__qa' in the filename")
    parser.add_argument("--glob", help="Custom glob pattern (e.g. '*__R.png')")
    parser.add_argument("--limit", type=int, help="Only process the first N images")
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Delay between calls in seconds (default: 0.8)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature (default: 0.6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without calling any model")
    parser.add_argument("--no-resume", action="store_true",
                        help="Process all images even if .txt already exists")

    args = parser.parse_args()

    # Track whether XAI_API_KEY was already present before loading .env files
    key_existed_before_env = "XAI_API_KEY" in os.environ

    # Load .env files (in order of priority)
    # 1. Current working directory (most common when running from the project root)
    # 2. Directory where the script lives
    # 3. The images folder passed via --folder
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    folder = resolve_cli_path(args.folder)

    loaded_env = load_env_file(cwd / ".env") or load_env_file(script_dir / ".env")
    if not loaded_env and folder.exists():
        load_env_file(folder / ".env")

    if not folder.exists():
        print_folder_error(args.folder, folder)
        sys.exit(1)
    if not folder.is_dir():
        print("ERROR: --folder must point to a directory.")
        print(f"  Input:    {args.folder}")
        print(f"  Resolved: {folder}")
        sys.exit(1)

    # Resolve vLLM URL early (with auto-detection support)
    vllm_base_url = None
    if args.backend == "vllm":
        vllm_base_url = get_vllm_base_url(args.vllm_ip, args.vllm_url)

    # Determine model (with interactive wizard for Ollama / vLLM when no --model given)
    if args.backend == "grok":
        model = args.model or "grok-4.3"
        if not HAS_OPENAI:
            print("ERROR: 'openai' package is required for Grok backend. Run: pip install openai")
            sys.exit(1)
    elif args.backend == "ollama":
        if args.model:
            model = args.model
        else:
            model = select_model_wizard("ollama", args.ollama_host, None)
        print(f"Using Ollama model: {model} @ {args.ollama_host}")
    else:  # vllm
        if args.model:
            model = args.model
        else:
            model = select_model_wizard("vllm", args.ollama_host, vllm_base_url)
        print(f"Using vLLM model: {model} @ {vllm_base_url}")

    # Build final system prompt
    if args.system_prompt_file:
        base_prompt = resolve_cli_path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    else:
        base_prompt = DEFAULT_SYSTEM_PROMPT

    if args.system_add:
        final_system_prompt = base_prompt.rstrip() + "\n\n" + args.system_add.strip()
    else:
        final_system_prompt = base_prompt

    # Get API key for Grok
    api_key = args.api_key or os.environ.get("XAI_API_KEY")

    # Determine where the API key came from (for nice startup message)
    if args.api_key:
        api_key_source = "command line (--api-key)"
    elif "XAI_API_KEY" in os.environ:
        if key_existed_before_env:
            api_key_source = "environment variable"
        else:
            api_key_source = ".env file"
    else:
        api_key_source = None

    if args.backend == "grok":
        if not api_key and not args.dry_run:
            print("ERROR: No XAI_API_KEY found for Grok backend.")
            print("")
            print("How to set your API key:")
            print("  1. Copy .env.example → .env")
            print("  2. Edit .env and add your key: XAI_API_KEY=sk-...")
            print("  3. Or set it as environment variable: export XAI_API_KEY=your_key")
            print("  4. Or pass it directly: --api-key your_key")
            print("")
            print("Get your key here: https://console.x.ai/")
            sys.exit(1)

        # Nice startup message for Grok users
        if not args.dry_run:
            print(f"✓ Using Grok backend (API key loaded from {api_key_source})")

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(300.0),
            max_retries=2,
        ) if not args.dry_run else None
    elif args.backend == "vllm":
        # vLLM uses OpenAI-compatible API. No real key needed.
        client = OpenAI(
            api_key="EMPTY",
            base_url=vllm_base_url,
            timeout=httpx.Timeout(300.0),
        ) if not args.dry_run else None
    else:
        client = None  # Ollama uses direct requests

    # Find images
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    if args.glob:
        images = sorted([p for p in folder.glob(args.glob) if p.suffix.lower() in exts])
    else:
        images = sorted([p for p in folder.iterdir()
                         if p.is_file() and p.suffix.lower() in exts])

    if args.skip_qa:
        images = [p for p in images if "__qa" not in p.name]

    if args.limit:
        images = images[:args.limit]

    if not images:
        print("No images found matching the criteria.")
        return

    max_side = args.max_side if args.max_side > 0 else None

    if args.backend == "vllm":
        print(f"Backend: VLLM | Model: {model} | URL: {vllm_base_url}")
    elif args.backend == "ollama":
        print(f"Backend: OLLAMA | Model: {model} | Host: {args.ollama_host}")
    else:
        print(f"Backend: GROK | Model: {model}")
    print(f"Found {len(images)} image(s) in {folder}")
    if args.skip_qa:
        print("  (skipping __qa* files)")
    if args.system_add:
        print(f"  (using additional system instructions)")
    if args.dry_run:
        print("  [DRY RUN - no API calls will be made]\n")

    processed = skipped = 0
    failed = []

    for idx, img_path in enumerate(images, 1):
        txt_path = img_path.with_suffix(".txt")
        status = f"[{idx}/{len(images)}] {img_path.name}"

        if args.dry_run:
            if txt_path.exists() and not (args.overwrite or args.no_resume):
                print(f"(skip - already has .txt) {status}")
            else:
                print(f"(will process) {status}")
            continue

        # Real processing path
        if txt_path.exists() and not (args.overwrite or args.no_resume):
            skipped += 1
            continue

        print(f"Processing {status} ...", end=" ", flush=True)

        try:
            if args.backend in ("grok", "vllm"):
                caption = call_openai_compatible(
                    client, model, final_system_prompt, img_path, max_side, args.temperature
                )
            else:
                caption = call_ollama(
                    args.ollama_host, model, final_system_prompt, img_path, max_side, args.temperature
                )

            # Light cleanup
            if caption.startswith("```"):
                caption = caption.strip("`").strip()
            if caption.lower().startswith("caption:"):
                caption = caption.split(":", 1)[1].strip()

            txt_path.write_text(caption + "\n", encoding="utf-8")
            processed += 1
            print("✓ done")

        except Exception as e:
            failed.append((img_path.name, str(e)))
            print(f"✗ FAILED: {e}")

        time.sleep(args.delay)

    print("\n" + "=" * 55)
    if args.dry_run:
        print("DRY RUN complete. No files were modified.")
    else:
        print(f"Complete. Processed: {processed} | Skipped (existing .txt): {skipped}")
        if failed:
            print(f"Failed: {len(failed)}")
            for name, err in failed:
                print(f"  - {name}: {err}")
        print("Captions saved as .txt files next to the images.")


if __name__ == "__main__":
    main()
