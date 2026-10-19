import sys

import onnxruntime as ort


def main(path: str) -> None:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"== {path}")
    for i in sess.get_inputs():
        print(f"  in : {i.name:12} {i.type:20} {i.shape}")
    for o in sess.get_outputs():
        print(f"  out: {o.name:12} {o.type:20} {o.shape}")
    meta = sess.get_modelmeta().custom_metadata_map
    if meta:
        print(f"  meta: {meta}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
