from __future__ import annotations
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from typing import Optional
from pathlib import Path


class SiburDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            encoder: Optional[OneHotEncoder] = None,
            period: Optional[dict[str, str]] = None,
            task: str = 'train',
            seq_range: int = 13
            ) -> None:
        """period['start'] - period['end'] for train and test
        if period = None, dataset in inference phase

        task - train/valid/inference,
               train for random sequences
               valid for row[:-1] sequences
               inference for full sequences and target=0

        seq_range - range in month for sequence length.
        """
        super().__init__()
        data = self._add_region(data)
        self.agg_cols = ["material_code", "company_code", "country", "region",
                         "manager_code", "material_lvl1_name", "material_lvl2_name",
                         "material_lvl3_name", "contract_type",
                         'region_big']
        if period is not None:
            data = data[(data['date'] >= period['start'])
                        & (data['date'] < period['end'])]

        self.raw_data = data.copy()
        self.data = data.groupby(self.agg_cols + ["month"])["volume"].sum() \
            .unstack(fill_value=0)
        
        self.encoder = encoder
        self.task = task
        self._create_features()
        self.seq_range = seq_range

        if task == 'train':
            self.create_encoder(data)
            self._build_long_dataset(period=seq_range)
        else:
            self.encoder = encoder
            # get only last columns of the data
            self.data = self.data.iloc[:, -seq_range:]

            if self.data.shape[1] != seq_range:
                raise ValueError(f'Wrong shape of dataset {self.data.shape}')


    def _create_features(self) -> None:
        """Build statistics along all company by each month (global context)."""
        self.total = self.raw_data.groupby(['month'])['volume'].sum()

        self.categories = [
            "material_code", "company_code", "country", "region",
            "manager_code", "material_lvl1_name", "material_lvl2_name",
            "material_lvl3_name", "contract_type",
            'region_big'
            ]
        functions = {'sum': 'sum'}
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


    def create_encoder(self, data: pd.DataFrame) -> None:
        print('Creating ohe encoder')
        self.encoder = OneHotEncoder()

        data['month_'] = data['date'].dt.month
        self.encoder.fit(data[self.agg_cols + ['month_']].values)

        with open('ohe_encoder.pkl', 'wb') as f:
            pickle.dump(self.encoder, f)

        features = self.encoder.transform(data[self.agg_cols + ['month_']].values).shape[1]
        print('OHE_encoder created with', features, 'features')


    def __len__(self) -> int:
        return len(self.data)


    def _build_ts(self, row: pd.Series) -> np.array:
        """Build timeseries with global context."""
        values = row.values.reshape(-1, 1)
        total = self.total.loc[row.index].values.reshape(-1, 1)
        metadata = row.name

        timeser = np.concatenate([values, total], axis=1)

        for cat in self.subsets.keys():
            for func, sub in self.subsets[cat].items():
                # get category number from metadata
                cat_index = metadata[self.agg_cols.index(cat)]
                # get timeser from subset
                try:
                    timeser_part = sub.loc[cat_index, row.index].values.reshape(-1, 1)
                except KeyError:
                    timeser_part = np.zeros_like(row.values).reshape(-1, 1)

                timeser = np.concatenate([timeser, timeser_part], axis=1)
        return timeser


    def _build_long_dataset(self, period: int = 12) -> None:
        """We need more data! Vertical stack data with rolling window."""
        groups = []
        # store information about last month
        self.months = []
        self.denominator = self.data.shape[0]

        for i in range(self.data.shape[1] - period):
            subset = self.data.iloc[:, i:period + i]

            self.months.append(subset.columns[-1])
            subset.columns = np.arange(0, period)
            groups.append(subset)
        self.data = pd.concat(groups, axis=0)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        row = self.data.iloc[index]
        row = row.sort_index()

        if self.task == 'train':
            row.index = pd.date_range(
                start=self.months[index // self.denominator] - pd.offsets.MonthBegin(self.seq_range - 1),
                periods=self.seq_range, freq='MS'
                )

        if self.task in ['valid', 'train']:
            target = row.values[-1]
            row = row[:-1]
        else:
            target = 0
            # skip first month of inference row due to alignment
            row = row[1:]

        values = self._build_ts(row)
        values = torch.tensor(values, dtype=torch.float32)
        target = torch.tensor([target], dtype=torch.float32)

        next_month = row.index[-1] + pd.offsets.MonthBegin(1)
        vector = list(row.name) + [next_month.month]
        vector = self.encoder.transform([vector]).toarray().flatten()
        vector = torch.tensor(vector, dtype=torch.float32)
        return values, vector, target
    
    
    def _add_region(self, data: pd.DataFrame) -> pd.DataFrame:
        """How about some hardcoding?"""
        reg_dict = {'Литва': 'европа',
         'Китай': 'азия',
         'Казахстан': 'снг',
         'Россия': 'снг',
         'Италия': 'европа',
         'Белоруссия': 'снг',
         'Германия': 'европа',
         'Франция': 'европа',
         'Соед. Королев.': 'европа',
         'Узбекистан': 'снг',
         'Польша': 'европа',
         'Нидерланды': 'европа',
         'Украина': 'снг',
         'Финляндия': 'европа',
         'Сербия': 'европа',
         'Турция': 'средняя азия',
         'Молдавия': 'европа',
         'Венгрия': 'европа',
         'Бельгия': 'европа',
         'Швейцария': 'европа',
         'Швеция': 'европа',
         'Эстония': 'европа',
         'Чехия': 'европа',
         'Австрия': 'европа',
         'Киргизия': 'снг',
         'Дания': 'европа',
         'Таджикистан': 'снг',
         'Испания': 'европа',
         'Словакия': 'европа',
         'Индия': 'азия',
         'Атырауская обл.': 'снг',
         'Рязанская обл.': 'цфо',
         'Алтайский край': 'сфо',
         'Пермский край': 'пфо',
         'Нижегородская обл.': 'пфо',
         'Свердловская обл.': 'уфо',
         'Брестская обл.': 'снг',
         'Ростовская обл.': 'юфо',
         'Московская обл.': 'цфо',
         'Респ. Башкортостан': 'пфо',
         'Минская обл.': 'снг',
         'Волгоградская обл.': 'юфо',
         'Иркутская обл.': 'сфо',
         'Владимирская обл.': 'цфо',
         'Респ. Татарстан': 'пфо',
         'Воронежская обл.': 'цфо',
         'Респ. Мордовия': 'пфо',
         'г. Санкт-Петербург': 'сзфо',
         'Смоленская обл.': 'цфо',
         'Тверская обл.': 'цфо',
         'Оренбургская обл.': 'пфо',
         'Курская обл.': 'цфо',
         'Самарская обл.': 'пфо',
         'Челябинская обл.': 'уфо',
         'Тульская обл.': 'цфо',
         'Краснодарский край': 'юфо',
         'Томская обл.': 'сфо',
         'Карагандинская обл.': 'снг',
         'Ставропольский край': 'скфо',
         'Кемеровская обл.': 'сфо',
         'г. Москва': 'цфо',
         'Омская обл.': 'сфо',
         'Ярославская обл.': 'цфо',
         'Ленинградская обл.': 'сзфо',
         'Гомельская обл.': 'снг',
         'Калининградская обл.': 'сзфо',
         'Брянская обл.': 'цфо',
         'Респ. Удмуртия': 'пфо',
         'Новосибирская обл.': 'сфо',
         'Пензенская обл.': 'пфо',
         'Хабаровский край': 'дфо',
         'Саратовская обл.': 'пфо',
         'Орловская обл.': 'цфо',
         'Ханты-Мансийский а. о.': 'уфо',
         'Ульяновская обл.': 'пфо',
         'Красноярский край': 'сфо',
         'Кировская обл.': 'пфо',
         'г. Алма-Ата': 'снг',
         'Гродненская обл.': 'снг',
         'Могилевская обл.': 'снг',
         'Приморский край': 'дфо',
         'Псковская обл.': 'сзфо',
         'Калужская обл.': 'цфо',
         'Витебская обл.': 'снг',
         'Тюменская обл.': 'уфо',
         'Павлодарская обл.': 'снг',
         'Западно-Казахстанская обл.': 'снг',
         'Липецкая обл.': 'цфо',
         'Ивановская обл.': 'цфо',
         'Еврейская АО': 'дфо',
         'Вологодская обл.': 'сзфо',
         'Мангистауская обл.': 'снг',
         'Респ. Саха (Якутия)': 'дфо',
         'г. Минск': 'снг',
         'г. Нур-Султан': 'снг',
         'Белгородская обл.': 'цфо',
         'Чувашская респ.': 'пфо',
         'Респ. Дагестан': 'скфо',
         'Респ. Коми': 'сзфо',
         'Астраханская обл.': 'юфо',
         'Восточно-Казахстанская обл.': 'снг',
         'Респ. Хакасия': 'сфо',
         'Респ. Марий Эл': 'пфо'}
        
        data['region_big'] = data['country'].map(reg_dict)
        return data


def get_loader(
        df: pd.DataFrame,
        encoder_path: Optional[Path] = None,
        shuffle: bool = False,
        period: Optional[dict[str, str]] = None,
        num_workers: int = 0,
        task: str = 'train',
        batch_size: int = 1,
        seq_range: int = 10
        ) -> DataLoader:
    if task != 'train':
        with open(str(encoder_path), 'rb') as f:
            encoder = pickle.load(f)
    else:
        encoder = None

    dataset = SiburDataset(
        data=df,
        encoder=encoder,
        period=period,
        task=task,
        seq_range=seq_range
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        drop_last=(task=='train')
        )
    return dataloader


def test_dataset(path: Path) -> None:
    def show_res(X, vector, target):
        print(f'X shape = {X.shape}, vector shape = {vector.shape}, target shape = {target.shape}')
        print(X[0, -1, :])
        print(target)
        
    
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
        batch_size=8
        )
    for X, vector, target in tqdm(train_dataloader):
        pass

    print()
    print('TRAIN')
    show_res(X, vector, target)

    train_dataloader = get_loader(
        df,
        shuffle=False,
        period={
            'start': '2018-01-01',
            'end': '2020-07-01'
            },
        num_workers=0,
        task='valid',
        encoder_path='ohe_encoder.pkl',
        batch_size=8
        )
    for X, vector, target in tqdm(train_dataloader):
        pass

    print()
    print('VALID')
    show_res(X, vector, target)

    inf_dataloader = get_loader(
        df,
        encoder_path='ohe_encoder.pkl',
        shuffle=False,
        period=None,
        num_workers=0,
        task='inference',
        batch_size=8
        )

    for X, vector, target in tqdm(inf_dataloader):
        pass

    print()
    print('INFERENCE')
    show_res(X, vector, target)


if __name__ == '__main__':
    path = Path('sc2021_train_deals.csv')
    test_dataset(path)
