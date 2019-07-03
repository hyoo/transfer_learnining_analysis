import _pickle as pkl
from pathlib import Path
import os
import argparse
import pandas as pd


class UnoDataLoader():
    def __init__(self,
                 source='GDSC',
                 cv=0,
                 data_root='../Data',
                 cv_root='../CCL_10Fold_Partition',
                 file_name='CCL_PDM_TransferLearningData_rmFactor_0.0_ddNorm_std.pkl'):
        self.pkl_path = Path(os.path.join(data_root, file_name))
        self.source = source
        self.cv = cv
        self.cv_root = cv_root

    def load(self):
        if self.cv is None:
            cl_train, cl_val = None, None
        else:
            cl_train, cl_val = self.filter_cl()

        with open(self.pkl_path, 'rb') as pkl_file:
            # read response
            res = pkl.load(pkl_file)
            res = res.loc[res['SOURCE'] == self.source]
            if cl_train is not None:
                res_train = res[res['ccl_name'].isin(cl_train)]
                res_val = res[res['ccl_name'].isin(cl_val)]
            else:
                res_train = res
                res_val = None
            # load cl properties and filter by geneGE
            genomics = pkl.load(pkl_file)
            cols = [x if x.startswith('geneGE_') else None for x in genomics.columns.tolist()]
            cols = list(filter(lambda x: x is not None, cols))
            genomics = genomics[cols]
            # load drug descriptors
            drug = pkl.load(pkl_file)

        df_y_train = res_train.reset_index(drop=True)
        df_x_train_cl = df_y_train.merge(genomics, left_on='ccl_name', how='left', right_index=True)
        df_x_train_dr = df_y_train.merge(drug, left_on='ctrpDrugID', how='left', right_index=True)
        df_x_train_cl.drop(columns=['SOURCE', 'ccl_name', 'ctrpDrugID', 'area_under_curve', 'groupID'], inplace=True)
        df_x_train_dr.drop(columns=['SOURCE', 'ccl_name', 'ctrpDrugID', 'area_under_curve', 'groupID'], inplace=True)

        df_y_val = res_val.reset_index(drop=True)
        df_x_val_cl = df_y_val.merge(genomics, left_on='ccl_name', how='left', right_index=True)
        df_x_val_dr = df_y_val.merge(drug, left_on='ctrpDrugID', how='left', right_index=True)
        df_x_val_cl.drop(columns=['SOURCE', 'ccl_name', 'ctrpDrugID', 'area_under_curve', 'groupID'], inplace=True)
        df_x_val_dr.drop(columns=['SOURCE', 'ccl_name', 'ctrpDrugID', 'area_under_curve', 'groupID'], inplace=True)

        return (df_y_train, df_x_train_cl, df_x_train_dr), (df_y_val, df_x_val_cl, df_x_val_dr)

    def filter_cl(self):
        train = []
        for finename in ['TrainList.txt', 'ValList.txt']:
            path = Path(os.path.join(self.cv_root, self.source, 'cv_{}'.format(self.cv), finename))
            train += read_text_list(path)
        path = Path(os.path.join(self.cv_root, self.source, 'cv_{}'.format(self.cv), 'TestList.txt'))
        val = read_text_list(path)
        return train, val


def read_text_list(path):
    with open(path, 'r') as txt_list_file:
        return list(map(lambda x: x.strip(), txt_list_file.readlines()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='GDSC',
                        choices=['GDSC', 'CCLE', 'CTRP', 'NCI60', 'gCSI'],
                        help='Data Source')
    parser.add_argument('--cv', type=int, default=0,
                        help='partition number')

    args, unparsed = parser.parse_known_args()

    loader = UnoDataLoader(source=args.data_source, cv=args.cv)
    (df_y_train, df_x_train_cl, df_x_train_dr), (df_y_val, df_x_val_cl, df_x_val_dr) = loader.load()

    store = pd.HDFStore('{}.{}.h5'.format(args.data_source, args.cv), 'w', complevel=9, complib='blosc:snappy')
    store.put('y_train', df_y_train)
    store.put('x_train_cl', df_x_train_cl)
    store.put('x_train_dr', df_x_train_dr)
    store.put('y_val', df_y_val)
    store.put('x_val_cl', df_x_val_cl)
    store.put('x_val_dr', df_x_val_dr)
    store.close()
