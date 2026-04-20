from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "rednote-hilab/dots.mocr"


def _repo_dir(model_path: str) -> str:
    repo_dir = model_path.replace("/", "--")
    if not repo_dir.startswith("models--"):
        repo_dir = f"models--{repo_dir}"
    return repo_dir


def _patch_dots_ocr_config(config_path: Path) -> bool:
    if not config_path.exists():
        return False

    text = config_path.read_text(encoding="utf-8")
    original = text

    old_sig = "def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):"
    new_sig = (
        "def __init__(self, image_processor=None, tokenizer=None, video_processor=None, "
        "chat_template=None, **kwargs):"
    )
    old_super = "super().__init__(image_processor, tokenizer, chat_template=chat_template)"
    new_super = "super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)"

    if old_sig in text:
        text = text.replace(old_sig, new_sig)
    if old_super in text:
        text = text.replace(old_super, new_super)

    if "self.video_token" not in text:
        marker = (
            '        self.image_token_id = 151665 if not hasattr(tokenizer, "image_token_id") '
            "else tokenizer.image_token_id\n"
        )
        if marker in text:
            text = text.replace(
                marker,
                marker
                + '        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token\n'
                + '        self.video_token_id = 151656 if not hasattr(tokenizer, "video_token_id") else tokenizer.video_token_id\n',
            )

    if text != original:
        config_path.write_text(text, encoding="utf-8")
        return True
    return False


def ensure_dots_ocr_processor_compat(model_path: str) -> None:
    if not model_path.endswith("dots.ocr"):
        return

    repo_dir = _repo_dir(model_path)
    cache_roots = [
        PROJECT_ROOT / ".container-home" / ".cache" / "huggingface",
        Path.home() / ".container-home" / ".cache" / "huggingface",
        Path.home() / ".cache" / "huggingface",
    ]

    targets: set[Path] = set()

    for root in cache_roots:
        hub_dir = root / "hub" / repo_dir / "snapshots"
        if hub_dir.exists():
            for snap in hub_dir.iterdir():
                cfg = snap / "configuration_dots.py"
                if cfg.exists():
                    targets.add(cfg)

    for root in cache_roots:
        modules_dir = root / "modules" / "transformers_modules" / "rednote_hyphen_hilab" / "dots_dot_ocr"
        if modules_dir.exists():
            for rev in modules_dir.iterdir():
                cfg = rev / "configuration_dots.py"
                if cfg.exists():
                    targets.add(cfg)

    for cfg in sorted(targets):
        _patch_dots_ocr_config(cfg)
