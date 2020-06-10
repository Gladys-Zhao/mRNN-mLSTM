#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MRNN_fixD_cell(nn.RNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'K', 'bias']
    def __init__(self, input_size, hidden_size, output_size, K, bias=True):
        super(MRNN_fixD_cell, self).__init__(input_size, hidden_size, bias, num_chunks=1)
        self.K = K
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mm = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fm = nn.Linear(input_size, hidden_size, bias=False)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.xh = nn.Linear(input_size, hidden_size, bias=False)
        self.hz = nn.Linear(hidden_size, output_size, bias=True)
        self.mz = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, input, wd, hx=None):
        if hx is None:
            h_0 = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            m_0 = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            xs = torch.zeros(self.K - 1, input.size(0), self.input_size, dtype=input.dtype, device=input.device)
            hx = (h_0, m_0, xs)

        h_0 = hx[0]
        m_0 = hx[1]
        xs = hx[2]

        xx = torch.cat([xs, input.view(-1, input.size(0), input.size(1))], 0)
        f = torch.einsum('ijk,ik->ijk', [xx, wd]).sum(dim=0)
        m = F.tanh(self.mm(m_0) + self.fm(f))
        h = F.tanh(self.hh(h_0) + self.xh(input))
        z = F.tanh(self.hz(h) + self.mz(m))
        xs_out = xx[1:, :]
        return z, (h, m, xs_out)

class MRNN_cell(nn.RNNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'K', 'bias']
    def __init__(self, input_size, hidden_size, output_size, K, bias=True):
        super(MRNN_cell, self).__init__(input_size, hidden_size, bias, num_chunks=1)  # weight_ih, weight_hh
        self.K = K
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.dd = nn.Linear(input_size, input_size, bias=True)
        self.hd = nn.Linear(hidden_size, input_size, bias=False)
        self.md = nn.Linear(hidden_size, input_size, bias=False)
        self.xd = nn.Linear(input_size, input_size, bias=False)

        self.mm = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fm = nn.Linear(input_size, hidden_size, bias=False)

        self.hh = nn.Linear(hidden_size, hidden_size, bias=True)
        self.xh = nn.Linear(input_size, hidden_size, bias=False)

        self.hz = nn.Linear(hidden_size, output_size, bias=True)
        self.mz = nn.Linear(hidden_size, output_size, bias=False)

    def get_ws(self, d):
        K = self.K
        w = [1.] * (K + 1)
        for i in range(K):
            w[K - i - 1] = w[K - i] * (i - d) / (i + 1)
        return torch.cat(w[0:K])

    def filter_d(self, hc, d):
        w = torch.ones(self.K, d.size(0), d.size(1))
        batch_size = w.shape[1]
        hidden_size = w.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                w[:, sample, hidden] = self.get_ws(d[sample, hidden].view([1]))
        outputs = hc.mul(w).sum(dim=0)
        return outputs

    def forward(self, input, hx=None):
        if hx is None:
            h_0 = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            m_0 = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            d_0 = torch.zeros(input.size(0), self.input_size, dtype=input.dtype, device=input.device)
            xs = torch.zeros(self.K - 1, input.size(0), self.input_size, dtype=input.dtype, device=input.device)
            hx = (h_0, m_0, d_0, xs)

        h_0 = hx[0]
        m_0 = hx[1]
        d_0 = hx[2]
        xs = hx[3]

        # dynamic d
        d = 0.5 * F.sigmoid(self.dd(d_0) + self.hd(h_0) + self.md(m_0) + self.xd(input))

        xx = torch.cat([xs, input.view(-1, input.size(0), input.size(1))], 0)
        f = self.filter_d(xx, d)
        m = F.tanh(self.mm(m_0) + self.fm(f))
        h = F.tanh(self.hh(h_0) + self.xh(input))
        z = self.hz(h) + self.mz(m)
        xs_out = xx[1:, :]
        return z, (h, m, d, xs_out)

class MLSTM_fixD_cell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, K):
        super(MLSTM_fixD_cell, self).__init__()
        self.hidden_size = hidden_size
        self.K = K
        self.output_size = output_size

        self.cgate = nn.Linear(input_size + hidden_size, hidden_size)
        self.igate = nn.Linear(input_size + hidden_size, hidden_size)
        self.fgate = nn.Linear(input_size + 2*hidden_size, hidden_size)
        self.ogate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, sample, hidden, celltensor, w):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=sample.dtype,
                                 device=sample.device)
        if celltensor is None:
            celltensor = torch.zeros(self.K, batch_size, self.hidden_size, dtype=sample.dtype,
                                 device=sample.device)

        combined = torch.cat((sample, hidden), 1)
        first = torch.einsum('ijk,ik->ijk', [-celltensor, w]).sum(dim=0)
        i_gate = self.igate(combined)
        o_gate = self.ogate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.cgate(combined)
        c_tilde = self.tanh(c_tilde)

        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        hc = torch.cat([celltensor, cell.view([-1, cell.size(0), cell.size(1)])], 0)
        hc1 = hc[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, hc1

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))

class MLSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size, K, output_size):
        super(MLSTM_cell, self).__init__()
        self.hidden_size = hidden_size
        self.K = K
        self.output_size = output_size

        self.cgate = nn.Linear(input_size + hidden_size, hidden_size)
        self.igate = nn.Linear(input_size + hidden_size, hidden_size)
        self.fgate = nn.Linear(input_size + 2*hidden_size, hidden_size)
        self.ogate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_ws(self, d):
        w = [1.] * (self.K + 1)
        for i in range(0, self.K):
            w[self.K - i - 1] = w[self.K - i] * (i - d) / (i + 1)
        return torch.cat(w[0:self.K])

    def filter_d(self, celltensor, d):
        w = torch.ones(self.K, d.size(0), d.size(1), dtype=d.dtype,
                                 device=d.device)
        hidden_size = w.shape[2]
        batch_size = w.shape[1]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                w[:, batch, hidden] = self.get_ws(d[batch, hidden].view([1]))
        outputs = celltensor.mul(w).sum(dim=0)
        return outputs

    def forward(self, sample, hidden, celltensor, d_0):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=sample.dtype,
                                 device=sample.device)
        if celltensor is None:
            celltensor = torch.zeros(self.K, batch_size, self.hidden_size, dtype=sample.dtype,
                                 device=sample.device)
        if d_0 is None:
            d_0 = torch.zeros(batch_size, self.hidden_size, dtype=sample.dtype,
                                 device=sample.device)

        combined = torch.cat((sample, hidden), 1)
        combined_d = torch.cat((sample, hidden, d_0), 1)
        d = self.fgate(combined_d)
        d = self.sigmoid(d) * 0.5
        first = -self.filter_d(celltensor, d)
        i_gate = self.igate(combined)
        o_gate = self.ogate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.cgate(combined)
        c_tilde = self.tanh(c_tilde)

        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        hc = torch.cat([celltensor, cell.view([-1, cell.size(0), cell.size(1)])], 0)
        hc1 = hc[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, hc1, d

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))
