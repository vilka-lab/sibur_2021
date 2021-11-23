from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import pickle


class SiburDataset(Dataset):
    def __init__(self, data, encoder, period=None, task='train'):
        """period['start'] - period['end'] for train and test
        if period = None, dataset in inference phase
        
        task - train/valid/inference,
               train for random sequences
               valid for row[:-1] sequences
               inference for full sequences and target=0"""
        super().__init__()
        agg_cols = ["material_code", "company_code", "country", "region",
                    "manager_code", "material_lvl1_name", "material_lvl2_name",
                    "material_lvl3_name", "contract_type"]
        if period is not None:
            data = data[(data['date'] >= period['start'])
                        & (data['date'] < period['end'])]
        self.data = data.groupby(agg_cols + ["month"])["volume"].sum().unstack(fill_value=0)
        self.encoder = encoder
        self.task = task


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        row = self.data.iloc[index]
        row = row.sort_index()

        if self.task == 'valid':
            target = row[-1]
            row = row.iloc[:-1]
        elif self.task == 'train':
            # get left border of random slice
            num_rows = len(row)
            start = np.random.randint(0, num_rows - 7)
            row = row.iloc[start:]

            # get right border of random slice
            num_rows = len(row)
            end = np.random.randint(6, num_rows - 1)
            target = row.iloc[end]
            row = row.iloc[:end]
        else:
            target = 0

        vector = self.encoder.transform([row.name]).toarray().flatten()
        vector = torch.tensor(vector)

        result = []
        for ix, val in zip(row.index, row):
            month = F.one_hot(torch.tensor(ix.month), num_classes=12)
            new_val = np.array([val, ])
            new_val = new_val
            result.append(new_val)

        target = torch.tensor([target], dtype=torch.float32)
        result = torch.tensor(np.array(result), dtype=torch.float32)
        return result, vector, target


def get_loader(df, encoder_path, shuffle=False, period=None, num_workers=0,
               task='train'):
    with open(str(encoder_path), 'rb') as f:
        encoder = pickle.load(f)

    dataset = SiburDataset(
        data=df,
        encoder=encoder,
        period=period,
        task=task
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
        )
    return dataloader
