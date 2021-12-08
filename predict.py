import sys
from pathlib import Path
ABS_PATH = Path(__file__).parent.absolute()
sys.path.append(str(ABS_PATH))
from model import Ensemble
from dataset import get_loader
import pandas as pd


def load_model(model_weights: list[Path]) -> Ensemble:
    model = Ensemble(model_weights)
    return model


def predict(
        df: pd.DataFrame,
        month: pd.Timestamp,
        num_workers: int = 2
        ) -> pd.DataFrame:
    encoder_path = ABS_PATH.joinpath('ohe_encoder.pkl')
    dataloader = get_loader(
        df,
        encoder_path=encoder_path,
        shuffle=False,
        period=None,
        num_workers=num_workers,
        task='inference',
        batch_size=8
        )
    preds = MODEL.predict(dataloader)

    test = df.groupby(dataloader.dataset.agg_cols + ["month"])["volume"] \
        .sum() \
        .unstack(fill_value=0)
    test['prediction'] = preds

    preds_df = test['prediction'] \
        .reset_index() \
        .pivot_table(
            index=['material_code', 'company_code', 'country', 'region', 'manager_code'],
            aggfunc='sum'
            ) \
        .reset_index()
    return preds_df


weights_path = ABS_PATH.joinpath('experiment')
weights = [
    weights_path.joinpath('last.pth')
    ]
MODEL = load_model(weights)
