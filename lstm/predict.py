import sys
from pathlib import Path
ABS_PATH = Path(__file__).parent.absolute()
sys.path.append(str(ABS_PATH))
# import torch
from model import Ensemble
from dataset import get_loader


def load_model(model_weights):
    model = Ensemble(model_weights)
    return model


def predict(df, month, num_workers=2):
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

    agg_cols = ["material_code", "company_code", "country", "region",
                "manager_code", "material_lvl1_name", "material_lvl2_name",
                "material_lvl3_name", "contract_type", 'region_big']
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


weights_path = ABS_PATH.joinpath('experiment')
weights = [
    weights_path.joinpath('last.pth')
    ]
MODEL = load_model(weights)


# torch.backends.quantized.engine = 'qnnpack'
# MODEL = torch.quantization.quantize_dynamic(
#     MODEL, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
# )