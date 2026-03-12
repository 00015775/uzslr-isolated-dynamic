"""
check_onnx_shapes.py
Run this to print the exact input/output shapes of the DWPose ONNX models.

Usage:
  python check_onnx_shapes.py --weights_dir ./dwpose_weights
"""
import argparse
from pathlib import Path

def inspect(onnx_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"\n{'='*60}")
    print(f"Model: {onnx_path.name}")
    print("INPUTS:")
    for inp in sess.get_inputs():
        print(f"  name={inp.name}  shape={inp.shape}  dtype={inp.type}")
    print("OUTPUTS:")
    for out in sess.get_outputs():
        print(f"  name={out.name}  shape={out.shape}  dtype={out.type}")

parser = argparse.ArgumentParser()
parser.add_argument("--weights_dir", default="./dwpose_weights")
args = parser.parse_args()

d = Path(args.weights_dir)
for f in sorted(d.glob("*.onnx")):
    inspect(f)
