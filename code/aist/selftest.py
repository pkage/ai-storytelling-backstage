#!/usr/bin/env python3

"""
Smoke test for public APIs in aist.text and aist.image.

Runs each public function once and reports pass/fail without stopping on errors.
"""

import inspect
import traceback
from typing import Any, Callable

from PIL import Image

from . import image, text


def _describe_result(result: Any) -> str:
    if result is None:
        return "None"
    if isinstance(result, list):
        return f"list(len={len(result)})"
    return type(result).__name__


def _run(name: str, fn: Callable[..., Any], **kwargs: Any) -> tuple[bool, str]:
    try:
        result = fn(**kwargs
        return True, _describe_result(result)
    except Exception:
        return False, traceback.format_exc(limit=1).strip()


def self_test() -> int:
    # Small generated images for image APIs that need image inputs.
    src_image = Image.new("RGB", (128, 128), "white")
    mask_image = Image.new("RGB", (128, 128), "black")

    tests: list[tuple[str, Callable[..., Any], dict[str, Any]]] = [
        (
            "aist.text.summarization",
            text.summarization,
            {
                "text": "Machine learning helps systems learn patterns from data.",
                "model": "sshleifer/tiny-mbart",
                "max_length": 20,
                "min_length": 5,
                "do_sample": False,
                "accelerate": True,
                "seed": 1,
                "render": False,
            },
        ),
        (
            "aist.text.text_generation",
            text.text_generation,
            {
                "prompt": "Once upon a time",
                "model": "small",
                "max_length": 20,
                "num_return_sequences": 1,
                "accelerate": True,
                "seed": 1,
                "render": False,
            },
        ),
        (
            "aist.text.sentiment_analysis",
            text.sentiment_analysis,
            {
                "text": "I love this.",
                "accelerate": True,
                "seed": 1,
                "render": False,
            },
        ),
        (
            "aist.text.mask_filling",
            text.mask_filling,
            {
                "text": "Paris is the [MASK] of France.",
                "accelerate": True,
                "seed": 1,
                "render": False,
            },
        ),
        (
            "aist.text.question_answering",
            text.question_answering,
            {
                "question": "What is the capital of France?",
                "context": "France has many cities. Paris is its capital.",
                "accelerate": True,
                "seed": 1,
                "render": False,
            },
        ),
        (
            "aist.text.instruct",
            text.instruct,
            {
                "prompt": "Say hello in one short sentence.",
                "accelerate": True,
                "seed": 1,
                "render": False,
            },
        ),
        (
            "aist.image.image_generation",
            image.image_generation,
            {
                "prompt": "A small red square",
                "grid_size": 1,
                "model_size": "mini",
                "temperature": 1.0,
                "show_in_progress": True,
                "accelerate": False,
                "render": False,
                "seed": 1,
            },
        ),
        (
            "aist.image.stable_diffusion",
            image.stable_diffusion,
            {
                "prompt": "A line drawing of a cat",
                "accelerate": True,
                "rounds": 1,
                "dims": (128, 128),
                "unsafe": True,
                "seed": 1,
            },
        ),
        (
            "aist.image.stable_diffusion_img2img",
            image.stable_diffusion_img2img,
            {
                "image": src_image,
                "prompt": "Turn this into a watercolor",
                "accelerate": True,
                "rounds": 1,
                "dims": (128, 128),
                "strength": 0.5,
                "guidance_scale": 5,
                "unsafe": True,
                "seed": 1,
            },
        ),
        (
            "aist.image.stable_diffusion_inpaint",
            image.stable_diffusion_inpaint,
            {
                "image": src_image,
                "mask_image": mask_image,
                "prompt": "Add a blue circle",
                "accelerate": True,
                "rounds": 1,
                "dims": (128, 128),
                "strength": 0.5,
                "guidance_scale": 5,
                "unsafe": True,
                "seed": 1,
            },
        ),
        (
            "aist.image.image_caption",
            image.image_caption,
            {
                "image": src_image,
                "max_length": 8,
                "num_beams": 1,
                "accelerate": True,
                "render": False,
            },
        ),
    ]

    expected_text = {
        n for n, obj in inspect.getmembers(text, inspect.isfunction) if not n.startswith("_")
    }
    expected_image = {
        n for n, obj in inspect.getmembers(image, inspect.isfunction) if not n.startswith("_")
    }
    covered = {name.split(".")[-1] for name, _, _ in tests}
    missing = sorted((expected_text | expected_image) - covered)

    if missing:
        print("WARNING: Some public functions are not covered:")
        for name in missing:
            print(f"  - {name}")
        print()

    passed = 0
    failed = 0

    for name, fn, kwargs in tests:
        ok, details = _run(name, fn, **kwargs)
        if ok:
            passed += 1
            print(f"PASS {name}: {details}")
        else:
            failed += 1
            print(f"FAIL {name}: {details}")

    print()
    print(f"Done. Passed: {passed}, Failed: {failed}, Total: {len(tests)}")
    return 0 if failed == 0 else 1

