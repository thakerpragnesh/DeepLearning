import pickle

import torch

import utils


class ModelPruner:
    def __init__(self, model, batch_per_epoch, config):
        """
        Initialize model pruner.
        """
        self.weights_count = utils.count_parameters(model)
        self.itr = 1
        self.pruners = []
        for name, weight in model.named_parameters():
            if 'bias' not in name:
                self.pruners.append(AutomatedGradualPruner(name, weight, batch_per_epoch, config[name]))

    def step(self):
        """
        Used after each training step. Applies mask to every weight
        Also updates mask at regular intervals with self.freq frequency
        (different for each layer)
        """
        for pruner in self.pruners:
            pruner.apply_mask()
            pruner.update_mask(self.itr)
        self.itr += 1

    def log(self):
        """
        Return total model sparsity
        """
        count_pruned = sum(pruner.get_pruned_count() for pruner in self.pruners)
        return count_pruned / self.weights_count

    def save_plot_data(self, path):
        from pathlib import Path

        for pruner in self.pruners:
            folder = path/pruner.name
            folder.mkdir(parents=True, exist_ok=True)
            with open(folder/'itr.pkl', 'wb') as i, \
                 open(folder/'sparsity.pkl', 'wb') as s, \
                 open(folder/'count.pkl', 'wb') as c:
                pickle.dump(pruner.itr, i)
                pickle.dump(pruner.sparsity, s)
                pickle.dump(pruner.pruned_count, c)
            


class WeightPruner:
    def __init__(self, name, weight, batch_per_epoch, config):
        self.name = name
        self.mask = torch.ones_like(weight, requires_grad=False)
        self.weight = weight
        start_itr = (config['start_epoch'] - 1) * batch_per_epoch
        ramp_itr = (config['ramp_epoch'] - 1) * batch_per_epoch
        end_itr = (config['end_epoch'] - 1) * batch_per_epoch
        freq = config['freq']
        q = config['q']

        # calculate the start slope and get
        # ramp slope by multiplying it
        self.start_slope = (2 * q * freq) / (2 * (ramp_itr - start_itr) + 3 * (end_itr - ramp_itr))
        self.ramp_slope = self.start_slope * config['ramp_slope_mult']
        self.start_itr = start_itr
        self.ramp_itr = ramp_itr
        self.end_itr = end_itr
        self.freq = freq

        self.itr = []
        self.sparsity = []
        self.pruned_count = []

    def update_mask(self, current_itr):
        # update stats
        self.itr.append(current_itr)
        self.sparsity.append(self.get_sparsity())
        self.pruned_count.append(self.get_pruned_count())

        if current_itr > self.start_itr and current_itr < self.end_itr:
            if current_itr % self.freq == 0:
                if current_itr < self.ramp_itr:
                    th = self.start_slope * (current_itr - self.start_itr + 1) / self.freq
                else:
                    th = (self.start_slope * (self.ramp_itr - self.start_itr + 1) +
                          self.ramp_slope * (current_itr - self.ramp_itr + 1)) / self.freq
                self.mask = torch.gt(torch.abs(self.weight.data), th).type(self.weight.type())

    def apply_mask(self):
        self.weight.data = self.weight.data * self.mask

    def get_weights_count(self):
        return int(torch.sum(self.mask).item())

    def get_pruned_count(self):
        return self.mask.numel() - self.get_weights_count()

    def get_density(self):
        return self.get_weights_count() / self.mask.numel()

    def get_sparsity(self):
        return self.get_pruned_count() / self.mask.numel()



class AutomatedGradualPruner():
    """Fine-grained pruning with an AGP sparsity schedule.
    An automated gradual pruning algorithm that prunes the smallest magnitude
    weights to achieve a preset level of network sparsity.
    """
    def __init__(self, name,  weight, batch_per_epoch, config):
        self.initial_sparsity = config['initial_sparsity']
        self.final_sparsity = config['final_sparsity']
        self.name = name
        self.mask = torch.ones_like(weight, requires_grad=False)
        self.weight = weight
        start_itr = (config['start_epoch'] - 1) * batch_per_epoch
        end_itr = (config['end_epoch'] - 1) * batch_per_epoch
       
        self.start_itr = start_itr
        self.end_itr = end_itr
        self.freq = config['freq']

        assert self.final_sparsity >= self.initial_sparsity

        self.itr = []
        self.sparsity = []
        self.pruned_count = []



    def update_mask(self, current_itr):
        span = ((self.end_itr - self.start_itr - 1) // self.freq) * self.freq
        assert span > 0

        target_sparsity = (self.final_sparsity +
                           (self.initial_sparsity-self.final_sparsity) *
                           (1.0 - ((current_itr-self.start_itr)/span))**3)

        self._set_param_mask_by_sparsity_target(target_sparsity)

    def _set_param_mask_by_sparsity_target(self,desired_sparsity):
        with torch.no_grad():
            # partial sort
            
            bottomk, _ = torch.topk(self.weight.data.abs().view(-1),
                                    int(desired_sparsity * self.weight.data.numel()),
                                    largest=False,
                                    sorted=True)
            if(bottomk.data.numel()):
                threshold = bottomk.data[-1]  # This is the largest element from the group of elements that we prune away
            else:
                threshold = 0
            self.mask = torch.gt(torch.abs(self.weight.data), threshold).type(self.weight.type())
    
    def apply_mask(self):
        self.weight.data = self.weight.data * self.mask

    def get_weights_count(self):
        return int(torch.sum(self.mask).item())

    def get_pruned_count(self):
        return self.mask.numel() - self.get_weights_count()

    def get_density(self):
        return self.get_weights_count() / self.mask.numel()

    def get_sparsity(self):
        return self.get_pruned_count() / self.mask.numel()