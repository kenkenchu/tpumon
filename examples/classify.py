"""Edge TPU image classification demo.

Runs MobileNet V2 (bird) on the Coral Edge TPU and prints top-5 results.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter, load_delegate

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = MODELS_DIR / "mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
DEFAULT_LABELS = MODELS_DIR / "inat_bird_labels.txt"
DEFAULT_IMAGE = MODELS_DIR / "parrot.jpg"


def load_labels(path: Path) -> dict[int, str]:
    labels = {}
    with open(path) as f:
        for i, line in enumerate(f):
            labels[i] = line.strip()
    return labels


def classify(model: Path, labels_path: Path, image_path: Path, top_k: int = 5):
    # Load Edge TPU delegate
    delegate = load_delegate("libedgetpu.so.1")
    interpreter = Interpreter(str(model), experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Resize image to model input size
    height, width = input_details["shape"][1], input_details["shape"][2]
    img = Image.open(image_path).convert("RGB").resize((width, height))
    input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

    # Run inference
    interpreter.set_tensor(input_details["index"], input_data)

    start = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Read results
    output = interpreter.get_tensor(output_details["index"]).squeeze()
    ordered = output.argsort()[::-1][:top_k]

    labels = load_labels(labels_path)

    print(f"Image:     {image_path.name}")
    print(f"Inference: {elapsed_ms:.1f} ms")
    print(f"Top-{top_k} results:")
    for i in ordered:
        score = output[i] / 255.0
        print(f"  {score:5.1%}  {labels.get(i, i)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge TPU classification demo")
    parser.add_argument("-m", "--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("-l", "--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("-i", "--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    classify(args.model, args.labels, args.image, args.top_k)
