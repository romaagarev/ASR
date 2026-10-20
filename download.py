import argparse
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path
import gdown


def download_file(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if "drive.google.com" in url:
        gdown.download(url, str(path), fuzzy=True)
    else:
        urllib.request.urlretrieve(url, path)


def download_example(output):
    import pyarrow.parquet as parquet

    url = (
        "https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/"
        "resolve/main/clean/validation-00000-of-00001.parquet"
    )
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "example.parquet"
        download_file(url, path)
        examples = parquet.read_table(path)[:3].to_pylist()

    (output / "audio").mkdir(parents=True, exist_ok=True)
    (output / "transcriptions").mkdir(parents=True, exist_ok=True)
    for example in examples:
        name = example["id"]
        (output / "audio" / f"{name}.flac").write_bytes(example["audio"]["bytes"])
        (output / "transcriptions" / f"{name}.txt").write_text(example["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--example", action="store_true")
    parser.add_argument("--unpack", action="store_true")
    args = parser.parse_args()

    if args.example:
        download_example(args.output)
    elif args.unpack:
        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "dataset"
            download_file(args.url, archive)
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as data:
                    data.extractall(args.output)
            else:
                with tarfile.open(archive) as data:
                    data.extractall(args.output, filter="data")
    else:
        download_file(args.url, args.output)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
