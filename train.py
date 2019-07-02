from model import build_model
# from data import UnoDataLoader
import pandas as pd

def run_cv_training(cv, dropout):
    (df_y_train, df_x_train_cl, df_x_train_dr), (df_y_val, df_x_val_cl, df_x_val_dr) = load_data()
    feature_shapes = {'cell.geneGE': (1927,), 'drug.dd': (4392,)}
    input_features = {'cell.geneGE': 'cell.geneGE', 'drug.dd': 'drug.dd'}
    args = {'drop': 0,
            'activation': 'relu',
            'dense_feature_layers': [1000, 1000, 1000],
            'dense': [1000, 1000, 1000],
            'residual': False}
    model = build_model(feature_shapes, input_features, args)
    model.summary()
    model.compile(loss='mse', optimizer='adam')
    model.fit([df_x_train_cl, df_x_train_dr], df_y_train,
              batch_size=512,
              epochs=50,
              validation_data=([df_x_val_cl, df_x_val_dr], df_y_val)
              )

def load_data():
    store = pd.HDFStore('cv.h5', 'r')
    df_y_train = store.get('y_train')
    df_x_train_cl = store.get('x_train_cl')
    df_x_train_dr = store.get('x_train_dr')
    df_y_val = store.get('y_val')
    df_x_val_cl = store.get('x_val_cl')
    df_x_val_dr = store.get('x_val_dr')

    return (df_y_train, df_x_train_cl, df_x_train_dr), (df_y_val, df_x_val_cl, df_x_val_dr)

def main():
    for cv in range(0, 9):
        for dropout in range(0, 0.5, 0.1):
            run_cv_training(cv, dropout)

if __name__ == '__main__':
    # main()
    run_cv_training(0, 0.1)
