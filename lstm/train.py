from model import SiburModel
import pathlib
from dataset import get_loader
import click
import torch
import pandas as pd


@click.command()
@click.option('--data', help='Path to train data', default='../sc2021_train_deals.csv')
@click.option('--model_weights', help='Path to saved model', default='./experiment/last.pth')
@click.option('--lr', help='Learning rate', default=1e-3)
@click.option('--weight_decay', default = 5e-3)
@click.option('--epochs', help='Number of epochs', default=30)
@click.option('--resume/--no-resume', help='Resume training process', default=False)
@click.option('--num_workers', help='Number of workers', default=2)
@click.option('--random_state', default=42)
def main(
        data: str,
        model_weights: str,
        lr: float,
        weight_decay: float,
        epochs: int,
        resume: bool,
        num_workers: int,
        random_state: int
        ):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    model = SiburModel(seed=random_state)
    model_path = pathlib.Path(model_weights)
    if model_path.exists():
        model.load_model(model_path, load_train_info=resume)
        print('Модель загружена с', model_path)

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

    df = pd.read_csv(data, parse_dates=["month", "date"])
    
    df['region_big'] = df['country'].map(reg_dict)
    
    train_dataloader = get_loader(
        df,
        shuffle=True,
        period={
            'start': '2018-01-01',
            'end': '2020-07-01'
            },
        num_workers=num_workers,
        task='train',
        batch_size=8
        )

    valid_dataloader = get_loader(
        df,
        shuffle=False,
        period={
            'start': '2018-01-01',
            'end': '2020-08-01'
            },
        num_workers=num_workers,
        task='valid',
        encoder_path='ohe_encoder.pkl',
        batch_size=8
        )

    model.fit(
        num_epochs=epochs,
        train_loader=train_dataloader,
        val_loader=valid_dataloader,
        folder='experiment',
        learning_rate=lr,
        weight_decay=weight_decay
        )


if __name__ == "__main__":
    main()
