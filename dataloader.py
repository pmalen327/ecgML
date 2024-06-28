# data loader for the ptbxl dataset 
# found here https://www.physionet.org/content/ptb-xl/1.0.3/example_physionet.py

import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = '/Users/Stone/source/repos/ecgML/ptbxl/'
sampling_rate=100
# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
# test_fold = 10
# Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

# X_train = X_train.transpose(2, 0, 1)
# X_test = X_test.transpose(2, 0, 1)

# # reducing batch size
# X_train = X_train[:, ::10, :]
# X_test = X_test[:, ::10, :]

# y_train = y_train[0::10]
# y_test = y_test[0::10]

# print(type(y_train), type(y_test))

# this is very cursed but does exactly what I need it to do lol
# X_train = X_train.reshape(12*19601, 1000)
# X_test = X_test.reshape(12*2198, 1000)