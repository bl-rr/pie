#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


HF_REPO_LINE = re.compile(r'^(\s*hf_repo\s*=\s*")([^"]*)(".*)$')
# Match both:
#   device = "cuda:2"
#   device = ["cuda:2", "cuda:3"]
# while preserving trailing inline comments.
DEVICE_LINE = re.compile(r"^(\s*device\s*=\s*)(.+?)(\s*(?:#.*)?)$")
MODEL_SECTION_LINE = re.compile(r"^\s*\[\[model\]\]\s*$")
ANY_SECTION_LINE = re.compile(r"^\s*(\[\[.*\]\]|\[.*\])\s*$")


def _apply_section_defaults(
    updated: list[str],
    saw_hf_repo: bool,
    saw_device: bool,
    model_id: str,
    device: str | None,
) -> None:
    if not saw_hf_repo:
        updated.append(f'hf_repo = "{model_id}"\n')
    if device is not None and not saw_device:
        updated.append(f'device = "{device}"\n')


def update_model(config_path: Path, model_id: str, device: str | None = None) -> int:
    content = config_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    model_section_count = 0
    in_model_section = False
    saw_hf_repo = False
    saw_device = False
    updated: list[str] = []

    for line in lines:
        stripped = line.rstrip("\n")

        # Section transition: flush defaults for the previous model section.
        if ANY_SECTION_LINE.match(stripped):
            if in_model_section:
                _apply_section_defaults(
                    updated,
                    saw_hf_repo,
                    saw_device,
                    model_id,
                    device,
                )
            in_model_section = bool(MODEL_SECTION_LINE.match(stripped))
            if in_model_section:
                model_section_count += 1
                saw_hf_repo = False
                saw_device = False
            updated.append(line)
            continue

        match = HF_REPO_LINE.match(line.rstrip("\n"))
        if in_model_section and match:
            # De-duplicate repeated hf_repo keys within the same [[model]] block.
            if saw_hf_repo:
                continue
            newline = "\n" if line.endswith("\n") else ""
            updated.append(f'{match.group(1)}{model_id}{match.group(3)}{newline}')
            saw_hf_repo = True
            continue

        match = DEVICE_LINE.match(line.rstrip("\n"))
        if in_model_section and match:
            # De-duplicate repeated device keys within the same [[model]] block.
            if saw_device:
                continue
            saw_device = True
            if device is not None:
                newline = "\n" if line.endswith("\n") else ""
                updated.append(f'{match.group(1)}"{device}"{match.group(3)}{newline}')
            else:
                updated.append(line)
            continue

        updated.append(line)

    # End-of-file: flush defaults for trailing model section.
    if in_model_section:
        _apply_section_defaults(
            updated,
            saw_hf_repo,
            saw_device,
            model_id,
            device,
        )

    if model_section_count == 0:
        if updated and not updated[-1].endswith("\n"):
            updated[-1] += "\n"
        updated.append("\n[[model]]\n")
        updated.append(f'hf_repo = "{model_id}"\n')
        if device is not None:
            updated.append(f'device = "{device}"\n')
        model_section_count = 1

    config_path.write_text("".join(updated), encoding="utf-8")
    return model_section_count


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path.home() / ".pie-eval" / "config.toml",
        help="Path to Pie config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model ID to set in all [[model]] sections",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Optional device override to set in all [[model]] sections (e.g. "cuda:2")',
    )
    args = parser.parse_args()

    config_path = args.config.expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    updated_count = update_model(config_path, args.model, args.device)
    device_note = f", device={args.device}" if args.device is not None else ""
    print(
        f"Updated {updated_count} model section(s) in {config_path} to {args.model}{device_note}"
    )


if __name__ == "__main__":
    main()
