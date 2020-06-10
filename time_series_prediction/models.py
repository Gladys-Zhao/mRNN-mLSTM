#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers

# RNN model for time series prediction
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, h=None):
        samples = input_y
        rnn_out, last_rnn_hidden = self.rnn(samples, h)
        output = self.hidden2output(rnn_out.view(-1, self.hidden_size))
        return output.view(samples.shape[0], samples.shape[1], self.output_size), last_rnn_hidden

# LSTM model for time series prediction
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.in_size = input_size
        self.h_size = hidden_size
        self.out_size = output_size
        self.lstmcell = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input, hx=None):
        time_steps = input.shape[0]
        batch_size = input.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.out_size)

        for t in range(time_steps):
            if hx is None:
                h_0 = torch.zeros(batch_size, self.h_size)
                c_0 = torch.zeros(batch_size, self.h_size)
                hx = (h_0, c_0)
            else:
                h_0 = hx[0]
                c_0 = hx[1]
            h_0, c_0 = self.lstmcell(input[t, :], (h_0, c_0))
            outputs[t, :] = self.output(h_0)
        return outputs, (h_0, c_0)

# mRNN with fixed d for time series prediction
class MRNN_fixD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, K, bias=True):
        super(MRNN_fixD, self).__init__()
        self.K = K
        self.input_size = input_size
        self.output_size = output_size
        self.bd = Parameter(torch.Tensor(torch.zeros(1, input_size)), requires_grad=True)
        self.lmrnncell = layers.MRNN_fixD_cell(input_size, hidden_size, output_size, K)

    def get_ws(self, d):
        K = self.K
        w = [1.] * (K + 1)
        for i in range(K):
            w[K - i - 1] = w[K - i] * (i - d) / (i + 1)
        return torch.cat(w[0:K])

    def get_wd(self, d):
        w = torch.ones(self.K, 1, d.size(1), dtype=d.dtype, device=d.device)
        batch_size = w.shape[1]
        hidden_size = w.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                w[:, sample, hidden] = self.get_ws(d[0, hidden].view([1]))
        return w.squeeze(1)

    def forward(self, inputs, h=None):
        time_steps = inputs.size(0)
        self.d = 0.5 * F.sigmoid(self.bd)
        wd = self.get_wd(self.d)
        for t in range(time_steps):
            outputs, h = self.lmrnncell(inputs[t, :], wd, h)
        return outputs, h

# mRNN with dynamic d for time series prediction
class MRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, K, bias=True):
        super(MRNN, self).__init__()
        self.K = K
        self.input_size = input_size
        self.output_size = output_size
        self.lmrnncell = layers.MRNN_cell(input_size, hidden_size, output_size, K)

    def forward(self, inputs, h=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.output_size)
        for t in range(time_steps):
            outputs[t, :], h = self.lmrnncell(inputs[t, :], h)
        return outputs, h

# mLSTM with fixed d for time series prediction
class MLSTM_fixD(nn.Module):
    def __init__(self, input_size, hidden_size, K, output_size):
        super(MLSTM_fixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K
        self.bd = Parameter(torch.Tensor(torch.zeros(1, hidden_size)), requires_grad=True)
        self.output_size = output_size
        self.mlstmcell = layers.MLSTM_fixD_cell(self.input_size, self.hidden_size, self.output_size, self.K)
        self.sigmoid = nn.Sigmoid()

    def get_ws(self, d):
        K = self.K
        w = [1.] * (K + 1)
        for i in range(K):
            w[K - i - 1] = w[K - i] * (i - d) / (i + 1)
        return torch.cat(w[0:K])

    def get_wd(self, d):
        w = torch.ones(self.K, 1, d.size(1), dtype=d.dtype, device=d.device)
        batch_size = w.shape[1]
        hidden_size = w.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                w[:, sample, hidden] = self.get_ws(d[0, hidden].view([1]))
        return w.squeeze(1)

    def forward(self, inputs, hx=None):
        if hx == None:
            hidden = None
            hc = None
        else:
            hidden = hx[0]
            hc = hx[1]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size, dtype=inputs.dtype, device=inputs.device)
        self.d = 0.5 * F.sigmoid(self.bd)
        wd = self.get_wd(self.d)
        for t in range(time_steps):
            outputs[t, :], hidden, hc = self.mlstmcell(inputs[t, :], hidden, hc, wd)
        return outputs, (hidden, hc)


# mLSTM with dynamic d for time series prediction
class MLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, K, output_size):
        super(MLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K
        self.output_size = output_size
        self.mlstmcell = layers.MLSTM_cell(self.input_size, self.hidden_size, self.K, self.output_size)

    def forward(self, inputs, hx=None):
        if hx == None:
            hidden = None
            hc = None
            d = None
        else:
            hidden = hx[0]
            hc = hx[1]
            d = hx[2]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size, dtype=inputs.dtype, device=inputs.device)
        for t in range(time_steps):
            outputs[t, :], hidden, hc, d = self.mlstmcell(inputs[t, :], hidden, hc, d)
        return outputs, (hidden, hc, d)

