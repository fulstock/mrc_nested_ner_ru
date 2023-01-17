# encoding: utf-8

from torch.utils.data import Dataset

import random

class TruncateDataset(Dataset):
    """Truncate dataset to certain num"""
    def __init__(self, dataset: Dataset, max_num: int = 100):

        # print(max_num)

        self.dataset = dataset
        self.max_num = min(max_num, len(self.dataset))

    def __len__(self):
        return self.max_num

    def __getitem__(self, item):
        return self.dataset[item]

    # def __getattr__(self, item):
    #     """other dataset func"""
    #     return getattr(self.dataset, item)

class TruncateByTypeDataset(Dataset):

    def __init__(self, dataset, max_num_for_type):

        self.dataset = dataset
        self.max_num_for_type = max_num_for_type

        self.data_dict = dict()

        for data in self.dataset:
            label = data[-1]
            if label not in self.data_dict.keys():
                self.data_dict[label] = [data]
            else:
                self.data_dict[label].append(data)

        self.truncset = []
        for tag, tag_data in self.data_dict.items():
            random.shuffle(tag_data)
            self.truncset.extend(tag_data[:self.max_num_for_type])

        self.tags = set(self.data_dict.keys())

    def __len__(self):
        return self.max_num_for_type * len(self.tags)

    def __getitem__(self, item):
        return self.truncset[item]

    # def __getattr__(self, item):
    #     return getattr(self.truncset, item)
