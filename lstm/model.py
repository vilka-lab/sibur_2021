# -*- coding: utf-8 -*-

import torch
import pickle
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from sklearn.metrics import mean_squared_log_error
import torch.nn.functional as F


class SiburModel(torch.nn.Module):
    def __init__(self, device=None, seed=42):
        torch.manual_seed(seed)
        super().__init__()
        hidden_size = 128
        self.lstm = torch.nn.LSTM(12, hidden_size, batch_first=True,
                                  num_layers=1, bidirectional=False,
                                  dropout=0.5, bias=False)

        vector_size = 542
        vector_embedding = 64
        self.cat_linear = torch.nn.Linear(vector_size, vector_embedding, bias=False)

        linear_hidden_size = hidden_size + vector_embedding
        self.linear_1 = torch.nn.Linear(linear_hidden_size, linear_hidden_size*2,
                                        bias=False)
        self.linear_2 = torch.nn.Linear(linear_hidden_size*2, 1,
                                        bias=False)

        self.relu = torch.nn.ReLU()

        # self.apply(weights_init_uniform_rule)

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.to(self.device)
        self.epoch = 0


    def forward(self, X, vector):
        X = X.log1p() - 2.67
        lstm_out, hidden = self.lstm(X)
        vector = self.cat_linear(vector)

        X = torch.cat([lstm_out[:, -1, ...], vector], dim=1)
        X = self.relu(X)

        X = self.linear_1(X)
        X = self.relu(X)

        X = self.linear_2(X)
        X = self.relu(X)
        X = X.exp() - 1
        return X


    def save_model(self, path):
        path = str(path)
        # for correct save batch norm after validation step
        self.train()
        checkpoint = {
            'epoch': self.epoch,
            'model': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_sched': self.scheduler}
        torch.save(checkpoint, path)


    def load_model(self, path, load_train_info=False):
        path = str(path)

        if self.device == 'cpu':
            checkpoint = torch.load(path, map_location=torch.device(self.device))
        else:
            checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model'])

        if load_train_info:
            self.create_optimizer()
            self.epoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler = checkpoint['lr_sched']


    def load_file(self, filename):
        filename = str(filename)
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        except FileNotFoundError:
            self.save_file(filename, [])
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
        return obj


    def save_file(self, filename, obj):
        filename = str(filename)
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)


    def load_storage(self, path):
        obj = {}
        obj['train_losses'] = self.load_file(path.joinpath('train_losses'))
        obj['train_metrics'] = self.load_file(path.joinpath('train_metrics'))
        obj['test_losses'] = self.load_file(path.joinpath('test_losses'))
        obj['test_metrics'] = self.load_file(path.joinpath('test_metrics'))
        return obj


    def save_storage(self, path, obj):
        self.save_file(path.joinpath('train_losses'), obj['train_losses'])
        self.save_file(path.joinpath('train_metrics'), obj['train_metrics'])
        self.save_file(path.joinpath('test_losses'), obj['test_losses'])
        self.save_file(path.joinpath('test_metrics'), obj['test_metrics'])


    def save_each_period(self, path, period=60*60):
        now = datetime.now()
        path = str(path)

        def save_model():
            nonlocal now
            delta = datetime.now() - now
            if delta.total_seconds() > period:
                self.save_model(path)
                now = datetime.now()
        return save_model


    @staticmethod
    def metric(preds, gt):
        score = mean_squared_log_error(preds, gt) ** 0.5
        return score


    def create_optimizer(self, learning_rate=1e-3, weight_decay=0):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)


    def fit(self, num_epochs, train_loader, val_loader=None,
              folder='experiment', learning_rate=1e-3, weight_decay=0,
              save_period=60*60):
        if not Path(folder).exists():
            os.mkdir(folder)
        path = Path('./').joinpath(folder)
        storage = self.load_storage(path)

        model_path = path.joinpath('last.pth')

        self.saver = self.save_each_period(model_path, save_period)
        self.loss = RMSLE
        self.create_optimizer(learning_rate, weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()

        # in respect for --resume flag
        for i in range(num_epochs - self.epoch):
            self.epoch += 1
            self._train_loop(train_loader, storage=storage)

            if val_loader is not None:
                self._test_loop(val_loader, storage=storage)

            self.save_model(model_path)
            self.save_storage(path, storage)
            self.visualize(folder, storage)

            print('-'*30)
        return storage


    def _train_loop(self, train_loader, storage):
        self.train()
        losses = []
        predictions = []
        gt = []
        with tqdm(total=len(train_loader)) as progress_bar:
            for X, vector, y in train_loader:
                self.optimizer.zero_grad()

                X = X.to(self.device)
                vector = vector.to(self.device)
                y = y.to(self.device)

                with torch.cuda.amp.autocast():
                    preds = self.forward(X, vector)
                    loss_val = self.loss(preds, y)

                self.scaler.scale(loss_val).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                losses.append(loss_val.item())

                predictions.append(preds.cpu().detach().numpy())
                gt.append(y.cpu().detach().numpy())

                self.saver()

                loss_val = np.mean(losses)

                progress_bar.update()
                progress_bar.set_description('Epoch {}: loss = {:.4f}'.format(
                                             self.epoch, loss_val))

            # calculate f1
            predictions = np.concatenate(predictions)
            gt = np.concatenate(gt)
            metric_val = self.metric(predictions, gt)
            progress_bar.set_description('Epoch {}: loss = {:.4f}, metric = {:.4f}'.format(
                self.epoch, loss_val, metric_val))

        storage['train_losses'].append(loss_val)
        storage['train_metrics'].append(metric_val)


    def _test_loop(self, val_loader, storage):
        self.eval()
        predictions = []
        gt = []
        losses = []
        with tqdm(total=len(val_loader)) as progress_bar:
            with torch.no_grad():
                for X, vector, y in val_loader:
                    X = X.to(self.device)
                    vector = vector.to(self.device)
                    y = y.to(self.device)

                    preds = self.forward(X, vector)
                    loss_val = self.loss(preds, y)

                    losses.append(loss_val.cpu().detach().numpy())

                    predictions.append(preds.cpu().detach().numpy())
                    gt.append(y.cpu().detach().numpy())

                    loss_val = np.mean(losses)

                    progress_bar.update()
                    progress_bar.set_description('Validation: loss = {:.4f}'.format(loss_val))

                # calculate f1
                predictions = np.concatenate(predictions)
                gt = np.concatenate(gt)
                metric_val = self.metric(predictions, gt)
                progress_bar.set_description('Validation: loss = {:.4f}, metric = {:.4f}'.format(
                    loss_val, metric_val))

        self.scheduler.step(loss_val)
        storage['test_losses'].append(loss_val)
        storage['test_metrics'].append(metric_val)


    def visualize(self, folder, storage=None):
        path = Path('./').joinpath(folder)

        if storage is None:
            storage = self.load_storage(path)

        plt.style.use('fivethirtyeight')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(storage['train_losses'], c='y', label='train loss')
        axes[0].plot(storage['test_losses'], c='b', label='validation loss')
        axes[0].set_title('losses')
        axes[1].plot(storage['train_metrics'], c='y', label='train_metric')
        axes[1].plot(storage['test_metrics'], c='b', label='validation metric')
        axes[1].set_title('metrics')
        plt.legend()
        fig.savefig(path.joinpath('results.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)


    def predict(self, dataloader, verbose=True):
        self.eval()
        result = []
        with torch.no_grad():
            for X, vector, _ in tqdm(dataloader, disable=(not verbose)):
                X = X.to(self.device)
                vector = vector.to(self.device)
                preds = self.forward(X, vector)
                result.append(preds.cpu().detach().numpy())
        result = np.concatenate(result)
        return result


    def postprocessing(self, gt):
        self.rounder = Rounder(100)
        gt = [self.rounder.get_nearest_value(x) for x in gt]
        return np.array(gt)


class Ensemble():
    def __init__(self, model_paths, **params):
        self.models = []

        for path in model_paths:
            path = Path(path)
            if path.exists():
                model = SiburModel(**params)
                model.load_model(path)
                print('Модель загружена с', path)
                self.models.append(model)
            else:
                raise ValueError(f'Модель не найдена {path}')


    def predict(self, dataloader, verbose=True):
        res = []
        for model in self.models:
            pred = model.predict(dataloader, verbose)
            res.append(pred)
            print(pred.mean())

        res = sum(res) / len(res)
        res = np.array(res)
        print(res.mean())
        return res


def RMSLE(pred, gt):
    pred = torch.log(pred + 1)
    gt = torch.log(gt + 1)
    return F.mse_loss(pred, gt) ** 0.5
    # return (((torch.log(gt + 1) - torch.log(pred + 1)) ** 2) ** 0.5).mean()


class Rounder():
    def __init__(self, bins=100):
        self.values = np.linspace(0, 9, bins)


    def get_nearest_value(self, num):
        """get nearest log"""
        num = np.log1p(num)
        num = min(self.values, key=lambda x: abs(x - num))
        num = np.exp(num) - 1
        return num


def weights_init_uniform_rule(m):
    """https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch"""
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
