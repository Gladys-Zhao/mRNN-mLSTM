#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import create_dataset, to_torch
import time_series_prediction.models as models

# initialization
parser = argparse.ArgumentParser()
# In time series prediction scenario, There are four dataset: 'tree7', 'traffic', 'arfima', 'DJI'.
parser.add_argument('--dataset', type=str, default='traffic', help='The test dataset')
# In review classification scenario, RNN, LSTM, mRNN_fixD, mLSTM_fixD, mRNN, mLSTM are tested.
parser.add_argument('--algorithm', type=str, default='mLSTM_fixD', help='The test algorithm')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--hidden_size', type=int, default=1, help='Number of hidden units.')
parser.add_argument('--input_size', type=int, default=1, help='Number of input units, as for time series it is 1.')
parser.add_argument('--output_size', type=int, default=1, help='Number of output units, as for time series it is 1.')
parser.add_argument('--K', type=int, default=100, help='Truncate the infinite summation at lag K.')
parser.add_argument('--patience', type=int, default=100, help='Patience.')
args = parser.parse_args()
algorithm = args.algorithm
dataset = args.dataset
batch_size = 1
start = 0
end = 100

# read data
df = pd.read_csv('../data/time_series_prediction/' + dataset + '.csv')

# split train/val/test
if dataset == 'tree7':
    train_size = 2500
    validate_size = 1000
if dataset == 'DJI':
    train_size = 2500
    validate_size = 1500
if dataset == 'traffic':
    train_size = 1200
    validate_size = 200
if dataset == 'arfima':
    train_size = 2000
    validate_size = 1200
rmse_list = []
mae_list = []
for i in range(start, end):
    seed = i
    print('seed ----------------------------------', seed)
    y = np.array(df["x"])
    dataset = y.reshape(-1, 1)
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # use this function to prepare the dataset for modeling
    X, Y = create_dataset(dataset, look_back=1)

    # split into train and test sets
    train_x, train_y = X[0:train_size], Y[0:train_size]
    validate_x, validate_y = X[train_size:train_size + validate_size], Y[train_size:train_size + validate_size]
    test_x, test_y = X[train_size + validate_size:len(Y)], Y[train_size + validate_size:len(Y)]

    # reshape input to be [time steps,samples,features]
    train_x = np.reshape(train_x, (train_x.shape[0], batch_size, args.input_size))
    validate_x = np.reshape(validate_x, (validate_x.shape[0], batch_size , args.input_size))
    test_x = np.reshape(test_x, (test_x.shape[0], batch_size, args.input_size))
    train_y = np.reshape(train_y, (train_y.shape[0], batch_size, args.output_size))
    validate_y = np.reshape(validate_y, (validate_y.shape[0], batch_size, args.output_size))
    test_y = np.reshape(test_y, (test_y.shape[0], batch_size, args.output_size))

    torch.manual_seed(seed)
    # initialize model
    if algorithm == 'RNN':
        model = models.RNN(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size)
    elif algorithm == 'LSTM':
        model = models.LSTM(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size)
    elif algorithm == 'mRNN_fixD':
        model = models.MRNN_fixD(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, K=args.K)
    elif algorithm == 'mRNN':
        model = models.MRNN(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, K=args.K)
    elif algorithm == 'mLSTM_fixD':
        model = models.MLSTM_fixD(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, K=args.K)
    elif algorithm == 'mLSTM':
        model = models.MLSTM(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, K=args.K)
    else:
        print('Algorithm selection ERROR!!!')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_loss = np.infty
    best_train_loss = np.infty
    stop_criterion = 1e-5
    rec = np.zeros((args.epochs, 3))
    epoch = 0
    val_loss = -1
    train_loss = -1
    cnt = 0

    def train():
        model.train()
        optimizer.zero_grad()
        target = torch.from_numpy(train_y).float()
        output, hx = model(torch.from_numpy(train_x).float())
        with torch.no_grad():
            val_y, _ = model(torch.from_numpy(validate_x).float(), hx)
            target_val = torch.from_numpy(validate_y).float()
            val_loss = criterion(val_y, target_val)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        return loss, val_loss

    def compute_test(best_model):
        model = best_model
        train_predict, hx = model(to_torch(train_x))
        train_predict = train_predict.detach().numpy()
        val_predict, hx = model(to_torch(validate_x), hx)
        test_predict, _ = model(to_torch(test_x), hx)
        test_predict = test_predict.detach().numpy()
        # invert predictions
        test_predict_r = scaler.inverse_transform(test_predict[:, 0, :])
        test_y_r = scaler.inverse_transform(test_y[:, 0, :])
        # calculate error
        test_rmse = math.sqrt(mean_squared_error(test_y_r[:, 0], test_predict_r[:, 0]))
        test_mape = (abs((test_predict_r[:, 0] - test_y_r[:, 0]) / test_y_r[:, 0])).mean()
        test_mae = mean_absolute_error(test_predict_r[:, 0], test_y_r[:, 0])
        return test_rmse, test_mape, test_mae

    while epoch < args.epochs:
        _time = time.time()
        loss, val_loss = train()
        if (val_loss < best_loss):
            best_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
        # stop_criteria = abs(criterion(val_Y, target_val) - val_loss)
        if ((best_train_loss - loss) > stop_criterion):
            best_train_loss = loss
            cnt = 0
        else:
            cnt += 1
        if cnt == args.patience:
            break
        # save training records
        time_elapsed = time.time()-_time
        rec[epoch, :] = np.array([loss, val_loss, time_elapsed])
        print("epoch: {:2.0f} train_loss: {:2.5f} val_loss: {:2.5f}  time: {:2.1f}s".format(
            epoch, loss.item(), val_loss.item(), time_elapsed))
        epoch = epoch + 1

    # make predictions
    test_rmse, test_mape, test_mae = compute_test(best_model)

    rmse_list.append(test_rmse)
    mae_list.append(test_mae)
    print('RMSE:{}'.format(rmse_list))
    print('MAE:{}'.format(mae_list))