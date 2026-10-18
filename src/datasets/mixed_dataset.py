from torch.utils.data import Dataset


class MixedDataset(Dataset):
    def __init__(self, datasets, text_encoder=None, ratio: tuple[int, int] = (1, 1)):
        assert len(datasets) == 2
        assert len(ratio) == 2
        self.datasets = list(datasets)
        self.ratio = tuple(ratio)
        self.group_size = sum(self.ratio)
        self.target_sr = self.datasets[0].target_sr
        assert self.datasets[1].target_sr == self.target_sr

        if text_encoder is not None:
            for dataset in self.datasets:
                dataset.text_encoder = text_encoder

    def __len__(self):
        return self.group_size * max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, index):
        group = index // self.group_size
        position = index % self.group_size
        if position < self.ratio[0]:
            dataset = self.datasets[0]
            dataset_index = group * self.ratio[0] + position
        else:
            dataset = self.datasets[1]
            dataset_index = group * self.ratio[1] + position - self.ratio[0]
        return dataset[dataset_index % len(dataset)]
