import json
import re
from pathlib import Path

import torchaudio
from tqdm import tqdm

from datasets import Audio, load_dataset
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CommonVoiceDataset(BaseDataset):
    def __init__(self, split, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "dataset_common_voice_17_0"
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(exist_ok=True, parents=True)
        self._regex = re.compile("[^a-z ]")
        index = self._get_or_load_index(split)
        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, split):
        index_path = self._data_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            dataset = load_dataset(
                "fsicoli/common_voice_17_0",
                "en",
                cache_dir=self._data_dir,
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
            dataset = dataset.select_columns(["path", "audio", "sentence"])
            dataset = dataset.cast_column("audio", Audio(decode=False))
            audio_dir = self._data_dir / split
            audio_dir.mkdir(exist_ok=True)
            index = []
            for entry in tqdm(dataset):
                path = audio_dir / Path(entry["path"]).name
                audio_bytes = entry["audio"]["bytes"]
                if not path.exists() or path.stat().st_size != len(audio_bytes):
                    path.write_bytes(audio_bytes)
                entry["path"] = str(path.absolute().resolve())
                entry["text"] = self._regex.sub("", entry.get("sentence", "").lower())
                t_info = torchaudio.info(entry["path"])
                entry["audio_len"] = t_info.num_frames / t_info.sample_rate
                index.append(
                    {
                        "path": entry["path"],
                        "text": entry["text"],
                        "audio_len": entry["audio_len"],
                    }
                )
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index
