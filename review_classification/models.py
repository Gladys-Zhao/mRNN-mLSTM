#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers

# RNN model for review classification
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.rnn1 = nn.RNN(self.input_size, self.hidden_size, dropout=dropout)
        self.rnn2 = nn.LSTM(self.hidden_size, self.hidden_size, dropout=dropout)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, length, h=None):
        samples = self.dropout(input_y)
        rnn_out1, last_rnn_hidden = self.rnn1(samples)
        rnn_out1 = F.tanh(rnn_out1)
        rnn_out, last_rnn_hidden = self.rnn2(rnn_out1)
        rnn_out = F.tanh(rnn_out)
        rnn_out_sel = torch.cat([rnn_out[length[i], i].unsqueeze(0) for i in range(length.shape[0])], dim=0)
        output = self.hidden2output(rnn_out_sel)
        return F.log_softmax(output, dim=1)

# LSTM model for review classification
class LSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(LSTM, self).__init__()
        self.in_size = in_size
        self.h_size = hidden_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)
        self.lstmcell_1 = nn.LSTMCell(in_size, hidden_size)
        self.lstmcell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)

    def forward(self, input, length, hx1=None, cx1=None, hx2=None, cx2=None):
        time_steps = input.shape[0]
        batch_size = input.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.h_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        for t in range(time_steps):
            if hx1 is None:
                hx1 = torch.zeros(batch_size, self.h_size, dtype=input.dtype, device=input.device)
            if cx1 is None:
                cx1 = torch.zeros(batch_size, self.h_size, dtype=input.dtype, device=input.device)
            inputs = self.dropout(input[t, :])
            hx1, cx1 = self.lstmcell_1(inputs, (hx1, cx1))
            hx1 = F.tanh(hx1)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.h_size, dtype=input.dtype, device=input.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.h_size, dtype=input.dtype, device=input.device)
            hx2, cx2 = self.lstmcell_2(hx1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[t, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i in range(length.shape[0])], dim=0)
        logit = self.output(outputs_sel)
        return F.log_softmax(logit, dim=1)

# mRNN with fixed d for review classification
class MRNN_fixD(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, K, dropout, bias=True):
        super(MRNN_fixD, self).__init__()
        self.K = K
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.lmrnncell1 = layers.MRNN_fixD_cell(input_size, hidden_size, hidden_size, K)
        self.lstmcell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)
		
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

    def forward(self, inputs, length, h1=None, hx2=None, cx2=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        self.d1 = 0.5 * F.sigmoid(self.bd1)
        wd1 = self.get_wd(self.d1)

        for t in range(time_steps):
            temp = self.dropout(inputs[t, :])
            outputs1, h1 = self.lmrnncell1(temp, wd1, h1)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstmcell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[t, :] = hx2

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1)


# mLSTM with fixed d for review classification
class MLSTM_fixD(nn.Module):
    def __init__(self, input_size, hidden_size, K, output_size):
        super(MLSTM_fixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K
        self.bd = Parameter(torch.Tensor(torch.zeros(1, hidden_size)), requires_grad=True)
        self.output_size = output_size
        self.mlstmcell = layers.MLSTM_fixD_cell(self.input_size, self.hidden_size, self.hidden_size, self.K)
        self.lstmcell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)
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

    def forward(self, inputs, length, hidden=None, hc=None, hx2=None, cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        self.d = 0.5 * F.sigmoid(self.bd)
        wd = self.get_wd(self.d)
        for t in range(time_steps):
            outputs1, hidden, hc = self.mlstmcell(inputs[t, :], hidden, hc, wd)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstmcell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[t, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1)