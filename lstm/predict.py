from model import SiburModel
from pathlib import Path
from dataset import get_loader
import torch


def load_model(model_weights):
    model = SiburModel()
    model_path = Path(model_weights)
    if model_path.exists():
        model.load_model(model_path)
        print('Модель загружена с', model_path)
    else:
        raise ValueError(f'Модель не найдена {model_weights}')
    return model


def predict(df, month, num_workers=2):
    model = load_model('experiment/last.pth')
    model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.ReLU},  # a set of layers to dynamically quantize
        dtype=torch.qint8
        )

    dataloader = get_loader(
        df,
        shuffle=False,
        period=None,
        num_workers=num_workers
        )
    preds = model.predict(dataloader)

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
