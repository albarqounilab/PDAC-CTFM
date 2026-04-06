import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple

class ALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data, 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10) -> None:
        """
        Initialize ALA module adapted for dictionary-based datasets
        """
        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        
        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        
        # FIX for Subset and dictionary dataset
        indices = list(range(len(self.train_data)))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        subset_indices = indices[rand_idx:rand_idx+rand_num]
        subset = Subset(self.train_data, subset_indices)
        rand_loader = DataLoader(subset, self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        if self.layer_idx > 0:
            for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
                param.data = param_g.data.clone()
        else: # if layer_idx is 0, preserve all layers (which means weight learning on all layers)
            pass

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        if self.layer_idx > 0:
            params_p = params[-self.layer_idx:]
            params_gp = params_g[-self.layer_idx:]
            params_tp = params_t[-self.layer_idx:]
            
            # frozen the lower layers to reduce computational cost in Pytorch
            for param in params_t[:-self.layer_idx]:
                param.requires_grad = False
        else:
            params_p = params
            params_gp = params_g
            params_tp = params_t

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp,
                                                self.weights):
            param_t.data = param + (param_g - param) * weight

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for batch in rand_loader:
                x = batch['image'].to(self.device)
                y = batch['label'].long().to(self.device)
                
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # modify according to the local objective
                loss_value.backward()

                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p,
                                                        params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]),
                    '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
