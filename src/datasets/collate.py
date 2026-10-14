import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audios = [item["audio"].squeeze(0) for item in dataset_items]
    spectrograms = [
        item["spectrogram"].squeeze(0).transpose(0, 1) for item in dataset_items
    ]
    text_encoded = [item["text_encoded"].squeeze(0).long() for item in dataset_items]

    return {
        "audio": pad_sequence(audios, batch_first=True),
        "audio_length": torch.tensor(
            [audio.shape[0] for audio in audios], dtype=torch.long
        ),
        "spectrogram": pad_sequence(spectrograms, batch_first=True).transpose(1, 2),
        "spectrogram_length": torch.tensor(
            [spectrogram.shape[0] for spectrogram in spectrograms], dtype=torch.long
        ),
        "text_encoded": pad_sequence(text_encoded, batch_first=True),
        "text_encoded_length": torch.tensor(
            [text.shape[0] for text in text_encoded], dtype=torch.long
        ),
        "text": [item["text"] for item in dataset_items],
        "audio_path": [item["audio_path"] for item in dataset_items],
    }
