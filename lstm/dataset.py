from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle


class SiburDataset(Dataset):
    def __init__(self, data, encoder, period=None, train=False):
        """period['start'] - period['end'] for train and test
        if period = None, dataset in inference phase"""
        super().__init__()
        agg_cols = ["material_code", "company_code", "country", "region",
                    "manager_code", "material_lvl1_name", "material_lvl2_name",
                    "material_lvl3_name", "contract_type"]
        if period is not None:
            data = data[(data['date'] >= period['start'])
                        & (data['date'] < period['end'])]
        self.data = data.groupby(agg_cols + ["month"])["volume"].sum().unstack(fill_value=0)
        self.encoder = encoder
        self.train = train


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        row = self.data.iloc[index]
        row = row.sort_index()

        if self.train:
            target = row[-1]
            row = row.iloc[:-1]
        else:
            target = 0

        result = []
        for ix, val in zip(row.index, row):
            vector = list(row.name) + [ix.month]
            vector = self.encoder.transform([vector]).toarray().flatten()
            new_val = np.concatenate([vector, np.array([val])])
            new_val = new_val
            result.append(new_val)

        target = torch.tensor([target], dtype=torch.float32)
        result = torch.tensor(np.array(result), dtype=torch.float32)
        return result, target


def get_loader(df, encoder_path, shuffle=False, period=None, num_workers=0,
               train=False):
    with open(str(encoder_path), 'rb') as f:
        encoder = pickle.load(f)

    dataset = SiburDataset(
        data=df,
        encoder=encoder,
        period=period,
        train=train
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0)
        )
    return dataloader
