#!/usr/bin/env python3
"""Generate Arduino C headers from an int8 TFLite model and IMU scaler."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def format_bytes(data: bytes, columns: int = 12) -> str:
    rows = []
    for start in range(0, len(data), columns):
        chunk = data[start : start + columns]
        rows.append("  " + ", ".join(f"0x{value:02x}" for value in chunk) + ",")
    return "\n".join(rows)


def format_floats(values: np.ndarray) -> str:
    return ", ".join(f"{float(value):.9g}f" for value in values)


def inspect_model(model_path: Path) -> tuple[list[int], str]:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required to validate the model. Install the project's "
            "training requirements first."
        ) from exc

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    input_info = interpreter.get_input_details()[0]
    shape = [int(value) for value in input_info["shape"]]
    dtype = np.dtype(input_info["dtype"]).name
    return shape, dtype


def write_model_header(model_path: Path, output_dir: Path) -> None:
    shape, dtype = inspect_model(model_path)
    element_count = int(np.prod(shape))
    if dtype != "int8":
        raise SystemExit(f"Model input must be int8, found {dtype}.")
    if element_count != 128 * 6:
        raise SystemExit(
            f"Model input must contain 128 x 6 values, found shape {shape}."
        )

    model = model_path.read_bytes()
    header = f"""#pragma once
#include <cstddef>
#include <cstdint>

alignas(16) const unsigned char g_model_data[] = {{
{format_bytes(model)}
}};

const std::size_t g_model_data_len = {len(model)};
const bool g_model_is_placeholder = false;
"""
    (output_dir / "model_data.h").write_text(header, encoding="utf-8")


def write_scaler_header(scaler_path: Path, output_dir: Path) -> None:
    with np.load(scaler_path) as scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)

    if mean.shape != (6,) or std.shape != (6,):
        raise SystemExit(
            f"Scaler mean/std must each have shape (6,), found {mean.shape}/{std.shape}."
        )
    if np.any(std <= 0):
        raise SystemExit("Scaler standard deviations must all be positive.")

    header = f"""#pragma once

const float kImuMean[6] = {{{format_floats(mean)}}};
const float kImuStd[6] = {{{format_floats(std)}}};
const bool kScalerIsPlaceholder = false;
"""
    (output_dir / "imu_scaler.h").write_text(header, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scaler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    write_model_header(args.model, args.output)
    write_scaler_header(args.scaler, args.output)
    print(f"Generated Arduino headers in {args.output.resolve()}")


if __name__ == "__main__":
    main()

