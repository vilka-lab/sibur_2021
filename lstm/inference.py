from torchvision import transforms as T
from model import Ensemble
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import os
import click


def read_folder(folder):
    extensions = ['.png', '.jpg', 'jpeg']
    result = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if Path(file).suffix in extensions:
                fpath = Path(root).joinpath(file)
                result.append(fpath)
    return result


def load_dataset(inp, folder):
    if inp is not None:
        path = Path(inp)
        if not path.exists():
            raise ValueError(f'Dataset path doesnt exist in {path}')
        
        if path.suffix == '.xlsx':
            files = pd.read_excel(path)
        elif path.suffix == '.csv':
            files = pd.read_csv(path)
        else:
            raise ValueError(f'Unknown extension of {path}, should be .xlsx or .csv')
        files = files['full_path'].values
    elif folder is not None:
        files = read_folder(folder)
    else:
        raise ValueError('Dataset not specified')
    return files


def restore_classes(df):
    classes = {
        0: '1_Работает',
        1: '2_Спит',
        2: '3_Смотрит в смартфон',
        3: '4_Отсутствует'
        }
    df['label'] = df['class'].replace(classes)
    return df


def load_model(model_weights, model_name, image_size):
    if len(model_weights) == 0:
        raise ValueError('No model weights')
        
    for path in model_weights:
        path = Path(path)
        if not path.exists():
            raise ValueError(f'File doesnt exist: {path}')
               
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.Resize((image_size, image_size)),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    ensemble = Ensemble(model_weights, transforms, model_name=model_name,
                        pretrained=False)
    return ensemble
    

@click.command()
@click.option('--inp', help='Path to dataset file with data', default=None)
@click.option('--out', help='Path to output file in excel format', default='output.xlsx')
@click.option('--folder', help='Path to folder with data, if --inp not specified',
              default=None)
@click.option('--model_weights', '-w', help='Paths to saved models', multiple=True)
@click.option('--image_size', default=224)
@click.option('--model_name', help='Effnet model name', default='efficientnet-b4')
@click.option('--threshold', help='Entropy threshold for "bad" classes', default=1.5)
def main(
        inp: str,
        out: str,
        folder: str,
        model_weights: list,
        image_size: int,
        model_name: str,
        threshold: float
        ):
    
    ensemble = load_model(model_weights, model_name, image_size)
    
    files = load_dataset(inp, folder)
    result = []
    for file in tqdm(files):
        try:
            img = Image.open(file)
        except Exception as err:
            print(f'{file}: {err}')
            result.append(None)
            
        pred, entropy = ensemble.predict([img], verbose=False, threshold=threshold)
        result.append([file, pred[0], entropy[0]])
    
    result = pd.DataFrame(result, columns=['file', 'class', 'entropy'])
    result = restore_classes(result)
    result.to_excel(out, index=False)
    

if __name__ == '__main__':
    main()
