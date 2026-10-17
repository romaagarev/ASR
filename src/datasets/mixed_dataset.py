from torch.utils.data import Dataset


class MixedDataset(Dataset):
    def __init__(self, datasets, text_encoder=None):
        assert len(datasets) == 2
        self.datasets = list(datasets)
        self.target_sr = self.datasets[0].target_sr
        assert self.datasets[1].target_sr == self.target_sr

        if text_encoder is not None:
            for dataset in self.datasets:
                dataset.text_encoder = text_encoder

    def __len__(self):
        return 2 * max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        dataset = self.datasets[index % 2]
        return dataset[(index // 2) % len(dataset)]
