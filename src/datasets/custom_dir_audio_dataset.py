from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        transcription_dir = Path(transcription_dir) if transcription_dir else None
        for path in sorted(Path(audio_dir).iterdir()):
            if not path.is_file() or path.suffix.lower() not in [
                ".mp3",
                ".wav",
                ".flac",
                ".m4a",
            ]:
                continue

            audio_info = torchaudio.info(str(path))
            entry = {
                "path": str(path.resolve()),
                "text": "",
                "audio_len": audio_info.num_frames / audio_info.sample_rate,
            }
            if transcription_dir is not None:
                transc_path = transcription_dir / (path.stem + ".txt")
                if transc_path.is_file():
                    with transc_path.open(encoding="utf-8") as f:
                        entry["text"] = f.read().strip()
            data.append(entry)
        super().__init__(data, *args, **kwargs)
