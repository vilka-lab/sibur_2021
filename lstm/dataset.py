from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class SiburDataset(Dataset):
    def __init__(self, data, encoder=None, scaler=None, period=None, task='train'):
        """period['start'] - period['end'] for train and test
        if period = None, dataset in inference phase

        task - train/valid/inference,
               train for random sequences
               valid for row[:-1] sequences
               inference for full sequences and target=0"""
        super().__init__()
        self.agg_cols = ["material_code", "company_code", "country", "region",
                         "manager_code", "material_lvl1_name", "material_lvl2_name",
                         "material_lvl3_name", "contract_type"]
        if period is not None:
            data = data[(data['date'] >= period['start'])
                        & (data['date'] < period['end'])]

        self.raw_data = data
        self.data = data.groupby(self.agg_cols + ["month"])["volume"].sum().unstack(fill_value=0)
        self.encoder = encoder
        self.task = task
        self._create_features()

        if task == 'train':
            self.create_encoder(data)
            # self.create_scaler()
        else:
            self.encoder = encoder
            # self.scaler = scaler


    def _create_features(self):
        self.total = self.raw_data.groupby(['month'])['volume'].sum()

        self.categories = ["material_code", "company_code", "country", "region",
                         "manager_code", "material_lvl1_name", "material_lvl2_name",
                         "material_lvl3_name", "contract_type"]
        functions = {'sum': 'sum', 'mean': 'mean', 'var': 'var',
                     'min': 'min', 'max': 'max'}
        self.subsets = {}

        for cat in self.categories:
            self.subsets[cat] = {}
            for f_name, func in functions.items():
                subset = self.raw_data.pivot_table(
                    index=cat,
                    columns='month',
                    values='volume',
                    aggfunc=func,
                    fill_value=0
                    )
                self.subsets[cat][f_name] = subset


    def create_encoder(self, data):
        print('Creating ohe encoder')
        self.encoder = OneHotEncoder()

        data['month_'] = data['date'].dt.month
        self.encoder.fit(data[self.agg_cols + ['month_']])

        with open('ohe_encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder, f)

        features = self.encoder.transform(data[self.agg_cols + ['month_']]).shape[1]
        print('OHE_encoder created with', features, 'features')


    def create_scaler(self):
        print('Create scaler')
        result = []
        for ix in range(self.__len__()):
            timeser = self._build_ts(self.data.iloc[ix])
            result.append(timeser)
        result = np.concatenate(result)

        self.scaler = StandardScaler()
        self.scaler.fit(result)

        with open('standard_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print('Standard scaler created')


    def __len__(self):
        return len(self.data)


    def _build_ts(self, row):
        """Build timeseries with additional features from row."""
        values = row.values.reshape(-1, 1)
        total = self.total.loc[row.index].values.reshape(-1, 1)
        metadata = row.name

        timeser = np.concatenate([values, total], axis=1)

        for cat in self.subsets.keys():
            for func, sub in self.subsets[cat].items():
                # get right category number from metadata
                cat_index = metadata[self.agg_cols.index(cat)]
                # get timeser from subset
                try:
                    timeser_part = sub.loc[cat_index, row.index].values.reshape(-1, 1)
                except KeyError:
                    timeser_part = np.zeros_like(row.values).reshape(-1, 1)

                timeser = np.concatenate([timeser, timeser_part], axis=1)

        return timeser


    def __getitem__(self, index):
        row = self.data.iloc[index]
        row = row.sort_index()
        values = self._build_ts(row)

        if self.task == 'valid':
            target = values[-1, 0]
            values = values[:-1, :]
        elif self.task == 'train':
            # get left border of random slice
            min_period = 12
            num_rows = len(values)
            start = np.random.randint(0, num_rows - min_period - 1)
            values = values[start:, :]
            row = row[start:]

            # get right border of random slice
            num_rows = len(values)
            end = np.random.randint(min_period, num_rows - 1)
            target = values[end, 0]
            values = values[:end, :]
            row = row[:end]
        else:
            target = 0

        # get the month for which we should predict value
        next_month = row.index[-1] + pd.offsets.MonthBegin(1)
        vector = list(row.name) + [next_month.month]
        vector = self.encoder.transform([vector]).toarray().flatten()
        vector = torch.tensor(vector, dtype=torch.float32)

        values = torch.tensor(values, dtype=torch.float32)
        target = torch.tensor([target], dtype=torch.float32)
        return values, vector, target


def get_loader(df, encoder_path=None, scaler_path=None, shuffle=False,
               period=None, num_workers=0, task='train'):
    if task != 'train':
        with open(str(encoder_path), 'rb') as f:
            encoder = pickle.load(f)

        with open(str(scaler_path), 'rb') as f:
            scaler = pickle.load(f)
    else:
        encoder = None
        scaler = None

    dataset = SiburDataset(
        data=df,
        encoder=encoder,
        scaler=scaler,
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
        task='train'
        )
    for X, vector, target in tqdm(train_dataloader):
        pass

    print()
    print(f'X shape = {X.shape}, vector shape = {vector.shape}, target shape = {target.shape}')
    print(X[0, -1, :])
    print(target)


if __name__ == '__main__':
    path = '../sc2021_train_deals.csv'
    test_dataset(path)
