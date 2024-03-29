import argparse
import csv
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
from model import build_model, r2, mae
from data import load_data
import pandas as pd
import numpy as np


def get_optimizer(optimizer):
    if optimizer == 'SGD':
        return SGD(lr=0.05, momentum=0.9)
    else:
        return optimizer


def train(params):
    print('start training on {} data, cv:{}, dropout: {}'
          .format(params.get('data_source'), params.get('cv'), params.get('dropout_rate')))

    (df_y_train, df_x_train_cl, df_x_train_dr), (df_y_val, df_x_val_cl, df_x_val_dr) = load_data(params, from_file=True)
    feature_shapes = {'cell.geneGE': (1927,), 'drug.dd': (4392,)}
    input_features = {'cell.geneGE': 'cell.geneGE', 'drug.dd': 'drug.dd'}
    y_train = df_y_train['area_under_curve'].values
    y_val = df_y_val['area_under_curve'].values

    tensorboard = TensorBoard(log_dir='tb/{}_cv{}_dr{}'.format(params['data_source'], params['cv'], params['dropout_rate']))
    callbacks = [tensorboard]

    model = build_model(feature_shapes, input_features, params)
    model.summary()
    model.compile(loss=params.get('loss'), optimizer=get_optimizer(params.get('optimizer')), metrics=[mae, r2])
    history = model.fit([df_x_train_cl, df_x_train_dr], y_train,
                        batch_size=params.get('batch_size'),
                        epochs=params.get('epochs'),
                        callbacks=callbacks,
                        validation_data=([df_x_val_cl, df_x_val_dr], y_val)
                        )
    stat = {'val_loss': history.history['val_loss'][-1],
            'val_r2': history.history['val_r2'][-1],
            'val_mae': history.history['val_mae'][-1]}

    return stat


def run_cv(params):
    cv_stat = {'val_mae': [], 'val_r2': [], 'val_loss': []}
    keys = list(cv_stat.keys())
    for cv in range(0, 10):
        params['cv'] = cv
        stat = train(params)
        for key in keys:
            cv_stat[key].append(stat[key])

    stat = {}
    for key in keys:
        stat['{}_mean'.format(key)] = np.mean(cv_stat[key])
        stat['{}_std'.format(key)] = np.std(cv_stat[key])
        stat['{}_min'.format(key)] = np.min(cv_stat[key])
        stat['{}_max'.format(key)] = np.max(cv_stat[key])
    return stat


def main(params):
    data = []
    for dropout in range(0, 6, 1):
        params['dropout_rate'] = dropout * 0.1
        cv_stat = run_cv(params)
        print('for dropout rate {}'.format(params['dropout_rate']), cv_stat)
        d = {'drouput_rate': params['dropout_rate']}
        datum = {**p_args, **d}
        data.append(datum)

    with open('result.csv') as csvfile:
        writer = csv.DictWriter(csvfile)
        for row in data:
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='GDSC',
                        choices=['GDSC', 'CCLE', 'CTRP', 'NCI60', 'gCSI'],
                        help='Data Source')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs')
    parser.add_argument('--cv', type=int, default=None,
                        help='partition number')
    parser.add_argument('--dropout_rate', type=float, default=None,
                        help='dropout rate')
    parser.add_argument('--model', type=str, default='uno',
                        choices=['uno', 'attention', 'deeper'],
                        help='model type')

    args, unparsed = parser.parse_known_args()
    p_args = vars(args)

    d_args = {'batch_size': 512,
              'activation': 'relu',
              'optimizer': 'adam',
              'loss': 'mse',
              'dense_feature_layers': [1000, 1000, 1000],
              'dense': [1000, 1000, 1000]}

    if args.model == 'attention':
        m_args = {'attention': True}
    elif args.model == 'deeper':
        m_args = {'dense_feature_layers': [1000, 1000, 1000, 1000, 1000],
                  'dense': [2000, 2000, 2000]}
    else:
        m_args = {'residual': True}

    params = {**p_args, **d_args, **m_args}

    if args.cv is not None:
        if args.dropout_rate is None:
            params['dropout_rate'] = 0
        train(params)
    elif args.dropout_rate is not None:
        run_cv(params)
    else:
        main(params)
