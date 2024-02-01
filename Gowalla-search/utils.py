import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Weights(torch.nn.Module):

    def __init__(self, softmax_type, option):
        super().__init__()
        self.option_size = len(option)
        self.candidate = 1
        if option[-1] == 0: self.candidate = 1
        initial_deep = np.ones((self.option_size, self.candidate))/self.option_size
        self.deep_weights = torch.nn.Parameter(torch.from_numpy(initial_deep).float(), requires_grad=True)
        self.softmax_type = softmax_type
        self.tau = 1.0

    def forward(self,):
        if self.tau > 0.01:
            self.tau -= 0.00005
        # print(f'self.tau={round(self.tau, 5)}')

        if self.softmax_type == 0:
            return F.softmax(self.deep_weights, dim=0)
        elif self.softmax_type == 1:
            return F.softmax(self.deep_weights/self.tau, dim=0)
        elif self.softmax_type == 2:
            if self.candidate==1:
                return F.softmax(self.deep_weights, dim=0)*len(self.deep_weights)/2
            return F.gumbel_softmax(self.deep_weights, tau=self.tau, hard=False, dim=-1)
        else:
            print('No such softmax_type'); print('TAU={}'.format(TAU)); quit()


class Weights_layer(torch.nn.Module):

    def __init__(self, softmax_type, n_layers):
        super().__init__()
        self.candidate = 1
        initial_deep = np.ones((n_layers, self.candidate))/n_layers
        self.deep_weights_layer = torch.nn.Parameter(torch.from_numpy(initial_deep).float(), requires_grad=True)
        self.softmax_type = softmax_type
        self.tau = 1.0

    def forward(self,):
        if self.tau > 0.01:
            self.tau -= 0.00005
        # print(f'self.tau={round(self.tau, 5)}')

        if self.softmax_type == 0:
            return F.softmax(self.deep_weights_layer, dim=0)
        elif self.softmax_type == 1:
            return F.softmax(self.deep_weights_layer/self.tau, dim=0)
        elif self.softmax_type == 2:
            if self.candidate==1:
                return F.softmax(self.deep_weights_layer, dim=0)*len(self.deep_weights_layer)/2
            return F.gumbel_softmax(self.deep_weights_layer, tau=self.tau, hard=False, dim=-1)
        else:
            print('No such softmax_type'); print('TAU={}'.format(TAU)); quit()
