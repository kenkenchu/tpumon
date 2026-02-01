"""Edge TPU inference for market regime detection.

Loads the quantized TFLite model (Edge TPU or CPU), fetches recent market
data, and predicts the current regime (Bull / Bear / Sideways).

Run: `uv run python regime/infer.py -t SPY`
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

from regime.data import REGIME_LABELS, prepare_inference_input

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_EDGETPU = MODELS_DIR / "regime_model_quant_edgetpu.tflite"
DEFAULT_MODEL_CPU = MODELS_DIR / "regime_model_quant.tflite"
STATS_PATH = MODELS_DIR / "regime_feature_stats.json"


def load_feature_stats(path: Path) -> dict:
    """Load feature normalization stats saved during training."""
    with open(path) as f:
        raw = json.load(f)
    return {
        "mean": np.array(raw["mean"], dtype=np.float64),
        "std": np.array(raw["std"], dtype=np.float64),
    }


def infer(ticker: str, model_path: Path, use_edgetpu: bool) -> None:
    """Run regime inference for a single ticker."""
    # Load feature stats
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Feature stats not found at {STATS_PATH}. Run regime/train.py first."
        )
    feature_stats = load_feature_stats(STATS_PATH)

    # Load interpreter
    if use_edgetpu:
        delegate = load_delegate("libedgetpu.so.1")
        interpreter = Interpreter(str(model_path), experimental_delegates=[delegate])
    else:
        interpreter = Interpreter(str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Fetch and prepare input
    input_data, date_str = prepare_inference_input(ticker, feature_stats)

    # Quantize float input to uint8 using model's quantization parameters
    input_scale = input_details["quantization"][0]
    input_zero_point = input_details["quantization"][1]
    input_quantized = np.clip(
        np.round(input_data / input_scale + input_zero_point), 0, 255
    ).astype(np.uint8)

    # Run inference
    interpreter.set_tensor(input_details["index"], input_quantized)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Read and dequantize output
    raw_output = interpreter.get_tensor(output_details["index"]).squeeze()
    output_scale = output_details["quantization"][0]
    output_zero_point = output_details["quantization"][1]
    probabilities = (raw_output.astype(np.float32) - output_zero_point) * output_scale

    # Ensure probabilities are non-negative and sum to 1
    probabilities = np.maximum(probabilities, 0)
    prob_sum = probabilities.sum()
    if prob_sum > 0:
        probabilities = probabilities / prob_sum

    regime_idx = np.argmax(probabilities)
    regime_name = REGIME_LABELS[regime_idx]

    backend = "Edge TPU" if use_edgetpu else "CPU"
    print(f"Ticker:    {ticker}")
    print(f"Date:      {date_str}")
    print(f"Backend:   {backend}")
    print(f"Inference: {elapsed_ms:.1f} ms")
    print(f"Regime:    {regime_name}")
    for i in range(3):
        print(f"  {REGIME_LABELS[i]:>8s}: {probabilities[i]:5.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market regime detection (Edge TPU)")
    parser.add_argument("-t", "--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=None,
        help="Path to TFLite model (auto-selected if omitted)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU instead of Edge TPU",
    )
    args = parser.parse_args()

    use_edgetpu = not args.cpu

    if args.model:
        model_path = args.model
    elif use_edgetpu:
        model_path = DEFAULT_MODEL_EDGETPU
    else:
        model_path = DEFAULT_MODEL_CPU

    if not model_path.exists():
        if use_edgetpu and DEFAULT_MODEL_CPU.exists():
            print(f"Edge TPU model not found at {model_path}")
            print(f"Falling back to CPU model: {DEFAULT_MODEL_CPU}")
            model_path = DEFAULT_MODEL_CPU
            use_edgetpu = False
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}. Run regime/train.py first."
            )

    infer(args.ticker, model_path, use_edgetpu)
