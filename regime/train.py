"""Train a market regime detection CNN and export as int8 TFLite.

Requires TensorFlow: install with `uv pip install -e ".[train]"`
Run: `uv run python regime/train.py`
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from regime.data import REGIME_LABELS, WINDOW_SIZE, NUM_FEATURES, build_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_model() -> tf.keras.Model:
    """Build a small CNN with Edge TPU-compatible ops only.

    All ops: Conv2D, ReLU6, AveragePooling2D, Reshape, Dense, Softmax.
    """
    inp = tf.keras.Input(shape=(WINDOW_SIZE, NUM_FEATURES, 1), name="input")

    # Conv blocks - kernels (k, 1) convolve along time axis only
    x = tf.keras.layers.Conv2D(
        16, (5, 1), padding="same", name="conv1"
    )(inp)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_1")(x)

    x = tf.keras.layers.Conv2D(
        32, (5, 1), padding="same", name="conv2"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_2")(x)

    x = tf.keras.layers.Conv2D(
        32, (3, 1), padding="same", name="conv3"
    )(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name="relu6_3")(x)

    # Global pooling - use AveragePooling2D + Reshape as fallback for Edge TPU
    # compatibility (GlobalAveragePooling2D may produce unsupported MEAN op)
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(WINDOW_SIZE, NUM_FEATURES), name="avg_pool"
    )(x)
    x = tf.keras.layers.Reshape((32,), name="reshape")(x)

    out = tf.keras.layers.Dense(3, activation="softmax", name="output")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name="regime_cnn")


def train(ticker: str = "SPY", epochs: int = 100, batch_size: int = 64) -> dict:
    """Train the model and return training history + dataset info."""
    print(f"Fetching data for {ticker}...")
    ds = build_dataset(ticker)

    X_train, y_train = ds["X_train"], ds["y_train"]
    X_test, y_test = ds["X_test"], ds["y_test"]

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"Input shape: {X_train.shape}")

    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
    ]

    # Class weights to compensate for 25/50/25 split
    class_weight = {0: 2.0, 1: 1.0, 2: 2.0}

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {acc:.4f} (baseline: 33.3%)")

    return {
        "model": model,
        "dataset": ds,
        "history": history,
        "test_loss": loss,
        "test_accuracy": acc,
    }


def export_tflite(model: tf.keras.Model, ds: dict, output_dir: Path) -> Path:
    """Export int8 quantized TFLite model.

    Uses post-training integer quantization with representative dataset.
    Input/output dtype: uint8.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train = ds["X_train"]

    # Representative dataset for calibration (200 samples)
    num_cal = min(200, len(X_train))
    cal_indices = np.linspace(0, len(X_train) - 1, num_cal, dtype=int)

    def representative_dataset():
        for i in cal_indices:
            yield [X_train[i : i + 1]]

    # Use from_concrete_functions with a fixed batch=1 input signature.
    # from_saved_model produces dynamic batch dims (shape_signature [-1,...])
    # which the Edge TPU compiler rejects.
    input_shape = [1] + list(model.input_shape[1:])

    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
    def inference(x):
        return model(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [inference.get_concrete_function()], model
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    output_path = output_dir / "regime_model_quant.tflite"
    output_path.write_bytes(tflite_model)
    print(f"\nExported quantized TFLite model: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

    return output_path


def save_feature_stats(ds: dict, output_dir: Path) -> Path:
    """Save feature normalization stats for use at inference time."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = output_dir / "regime_feature_stats.json"
    stats = {
        "mean": ds["feature_stats"]["mean"].tolist(),
        "std": ds["feature_stats"]["std"].tolist(),
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Saved feature stats: {stats_path}")
    return stats_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train market regime CNN")
    parser.add_argument("-t", "--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    result = train(args.ticker, args.epochs, args.batch_size)

    model_path = export_tflite(result["model"], result["dataset"], MODELS_DIR)
    save_feature_stats(result["dataset"], MODELS_DIR)

    print(f"\nNext steps:")
    print(f"  1. Compile for Edge TPU (on x86_64):")
    print(f"     edgetpu_compiler -s {model_path} -o {MODELS_DIR}/")
    print(f"  2. Run inference:")
    print(f"     uv run python regime/infer.py -t {args.ticker}")
