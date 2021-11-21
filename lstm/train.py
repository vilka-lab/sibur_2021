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
@click.option('--weight_decay', default=0.)
@click.option('--epochs', help='Number of epochs', default=30)
@click.option('--resume/--no-resume', help='Resume training process', default=False)
@click.option('--num_workers', help='Number of workers', default=2)
def main(
        data: str,
        model_weights: str,
        lr: float,
        weight_decay: float,
        epochs: int,
        resume: bool,
        num_workers: int
        ):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)

    model = SiburModel(hidden_dim=2048, num_layers=2)
    model_path = pathlib.Path(model_weights)
    if model_path.exists():
        model.load_model(model_path, load_train_info=resume)
        print('Модель загружена с', model_path)


    df = pd.read_csv(data, parse_dates=["month", "date"])
    train_dataloader = get_loader(
        df,
        shuffle=True,
        period={
            'start': '2018-01-01',
            'end': '2019-12-01'
            },
        num_workers=num_workers
        )

    valid_dataloader = get_loader(
        df,
        shuffle=False,
        period={
            'start': '2019-12-01',
            'end': '2020-08-01'
            },
        num_workers=num_workers
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