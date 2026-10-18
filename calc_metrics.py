import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("target_dir", type=Path)
    parser.add_argument("prediction_dir", type=Path)
    args = parser.parse_args()

    targets = {path.name: path for path in args.target_dir.glob("*.txt")}
    predictions = {
        path.name: path for path in args.prediction_dir.glob("*.txt")
    }

    cer = 0.0
    wer = 0.0
    for name, path in sorted(targets.items()):
        target = CTCTextEncoder.normalize_text(path.read_text(encoding="utf-8").strip())
        prediction = CTCTextEncoder.normalize_text(
            predictions[name].read_text(encoding="utf-8").strip()
        )
        cer += calc_cer(target, prediction)
        wer += calc_wer(target, prediction)

    print(f"Samples: {len(targets)}")
    print(f"CER: {100 * cer / len(targets):.2f}%")
    print(f"WER: {100 * wer / len(targets):.2f}%")


if __name__ == "__main__":
    main()
