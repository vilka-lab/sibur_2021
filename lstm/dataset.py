from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm


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
            min_period = 12
            num_rows = len(row)
            start = np.random.randint(0, num_rows - min_period - 1)
            row = row.iloc[start:]

            # get right border of random slice
            num_rows = len(row)
            end = np.random.randint(min_period, num_rows - 1)
            target = row.iloc[end]
            row = row.iloc[:end]
        else:
            target = 0

        # get the month for which we should predict value
        next_month = row.index[-1] + pd.offsets.MonthBegin(1)  
        vector = list(row.name) + [next_month.month]
 
        vector = self.encoder.transform([vector]).toarray().flatten()
        vector = torch.tensor(vector, dtype=torch.float32)

        values = torch.tensor(row.values, dtype=torch.float32).reshape(-1, 1)      
        target = torch.tensor([target], dtype=torch.float32)
        return values, vector, target


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


def test_dataset(path):
    df = pd.read_csv(path, parse_dates=["month", "date"])
    train_dataloader = get_loader(
        df,
        shuffle=True,
        period={
            'start': '2018-01-01',
            'end': '2020-07-01'
            },
        num_workers=0,
        task='train',
        encoder_path='ohe_encoder.pkl'
        )
    for X, vector, target in tqdm(train_dataloader):
        pass
    
    print()
    print(f'X shape = {X.shape}, vector shape = {vector.shape}, target shape = {target.shape}')
    
    
if __name__ == '__main__':
    path = '../sc2021_train_deals.csv'
    test_dataset(path)
    