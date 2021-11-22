import sys
from pathlib import Path
ABS_PATH = Path(__file__).parent.absolute()
sys.path.append(str(ABS_PATH))
import torch
from model import SiburModel
from dataset import get_loader
import pandas as pd


def load_model(model_weights):
    model = SiburModel(hidden_dim=2048, num_layers=2)
    model_path = Path(model_weights)
    if model_path.exists():
        model.load_model(model_path, load_train_info=True)
        print('Модель загружена с', model_path)
    else:
        raise ValueError(f'Модель не найдена {model_weights}')
    return model


def predict(df, month, num_workers=2):
    border = df['month'].max() - pd.offsets.MonthBegin(6)
    df = df[df['date'] >= border]
    encoder_path = ABS_PATH.joinpath('ohe_encoder.pkl')

    dataloader = get_loader(
        df,
        encoder_path=encoder_path,
        shuffle=False,
        period=None,
        num_workers=num_workers
        )
    preds = MODEL.predict(dataloader)

    agg_cols = ["material_code", "company_code", "country", "region",
                "manager_code", "material_lvl1_name", "material_lvl2_name",
                "material_lvl3_name", "contract_type"]
    test = df.groupby(agg_cols + ["month"])["volume"].sum().unstack(fill_value=0)
    test['prediction'] = preds

    preds_df = test['prediction'] \
        .reset_index() \
        .pivot_table(
            index=['material_code', 'company_code', 'country', 'region', 'manager_code'],
            aggfunc='sum'
            ) \
        .reset_index()
    return preds_df


MODEL = load_model(ABS_PATH.joinpath('experiment', 'last.pth'))
MODEL = torch.quantization.quantize_dynamic(
    MODEL, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
)