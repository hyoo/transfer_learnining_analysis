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
    cv_stat = {'val_mae': [], 'val_r2': []}
    for cv in range(0, 10):
        params['cv'] = cv
        stat = train(params)
        cv_stat['val_r2'].append(stat['val_r2'])

    return {'mean': np.mean(cv_stat['val_r2']),
            'std': np.std(cv_stat['val_r2']),
            'min': np.min(cv_stat['val_r2']),
            'max': np.max(cv_stat['val_r2'])}


def main(params):
    for dropout in range(0, 6, 1):
        params['dropout_rate'] = dropout * 0.1
        cv_stat = run_cv(params)
        print('for dropout rate {}'.format(params['dropout_rate']), cv_stat)


if __name__ == '__main__':
    params = {'data_source': 'GDSC',
              'batch_size': 512,
              'epochs': 30,
              'activation': 'relu',
              'optimizer': 'adam',
              'loss': 'mse',
              'dense_feature_layers': [1000, 1000, 1000],
              'dense': [1000, 1000, 1000],
              'residual': False}

    # main(params)
    params['cv'] = 0
    params['dropout_rate'] = 0.5
    train(params)
    # params['dropout_rate'] = 0.1
    # stat = run_cv(params)
    # print(stat)
