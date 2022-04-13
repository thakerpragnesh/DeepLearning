import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from collections import namedtuple, OrderedDict
import logging
import utils

__all__ = ['PruningPolicy',  'LRPolicy', 'ScheduledTrainingPolicy',
           'PolicyLoss', 'LossComponent']

msglogger = logging.getLogger()
PolicyLoss = namedtuple('PolicyLoss', ['overall_loss', 'loss_components'])
LossComponent = namedtuple('LossComponent', ['name', 'value'])


class ScheduledTrainingPolicy(object):
    def __init__(self, classes=None, layers=None):
        self.classes = classes
        self.layers = layers

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        pass

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        pass

    def before_backward_pass(self, model, epoch, minibatch_id, minibatches_per_epoch, loss, zeros_mask_dict,
                             optimizer=None):
        pass

    def before_parameter_optimization(self, model, epoch, minibatch_id, minibatches_per_epoch,
                                      zeros_mask_dict, meta, optimizer):
        pass

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        """The mini-batch training pass has ended"""
        pass

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        """The current epoch has ended"""
        pass


class PruningPolicy(ScheduledTrainingPolicy):
    def __init__(self, pruner, pruner_args, classes=None, layers=None):
        super(PruningPolicy, self).__init__(classes, layers)
        self.pruner = pruner
        if pruner_args is None:
            pruner_args = {}
        self.levels = pruner_args.get('levels', None)
        self.keep_mask = pruner_args.get('keep_mask', False)
        self.mini_batch_pruning_frequency = pruner_args.get('mini_batch_pruning_frequency', 0)
        self.mask_on_forward_only = pruner_args.get('mask_on_forward_only', False)
        self.mask_gradients = pruner_args.get('mask_gradients', False)
        if self.mask_gradients and not self.mask_on_forward_only:
            raise ValueError("mask_gradients and (not mask_on_forward_only) are mutually exclusive")
        self.backward_hook_handle = None   # The backward-callback handle
        self.use_double_copies = pruner_args.get('use_double_copies', False)
        self.discard_masks_at_minibatch_end = pruner_args.get('discard_masks_at_minibatch_end', False)
        self.skip_first_minibatch = pruner_args.get('skip_first_minibatch', False)
        self.fold_bn = pruner_args.get('fold_batchnorm', False)
        self.named_modules = None
        self.sg = None
        self.is_last_epoch = False
        self.is_initialized = False

    @staticmethod
    def _fold_batchnorm(model, param_name, param, named_modules):
        def _get_all_parameters(param_module, bn_module):
            w, b, gamma, beta = param_module.weight, param_module.bias, bn_module.weight, bn_module.bias
            if not bn_module.affine:
                gamma = 1.
                beta = 0.
            return w, b, gamma, beta

        def get_bn_folded_weights(conv_module, bn_module):

            w, b, gamma, beta = _get_all_parameters(conv_module, bn_module)
            with torch.no_grad():
                sigma_running = torch.sqrt(bn_module.running_var + bn_module.eps)
                w_corrected = w * (gamma / sigma_running).view(-1, 1, 1, 1)
            return w_corrected

        layer_name = utils.param_name_2_module_name(param_name)
        if not isinstance(named_modules[layer_name], nn.Conv2d):
            return param

        bn_layers = sg.successors_f(layer_name, ['BatchNormalization'])
        if bn_layers:
            assert len(bn_layers) == 1
            bn_module = named_modules[bn_layers[0]]
            conv_module = named_modules[layer_name]
            param = get_bn_folded_weights(conv_module, bn_module)
        return param

    def on_epoch_begin(self, model, zeros_mask_dict, meta, **kwargs):
        msglogger.debug("Pruner {} is about to prune".format(self.pruner.name))
        self.is_last_epoch = meta['current_epoch'] == (meta['ending_epoch'] - 1)
        if self.levels is not None:
            self.pruner.levels = self.levels

        meta['model'] = model
        is_initialized = self.is_initialized

        if self.fold_bn:
            # Cache this information (required for BN-folding) to improve performance
            self.named_modules = OrderedDict(model.named_modules())
            dummy_input = torch.randn(model.input_shape)
            # self.sg = distiller.SummaryGraph(model, dummy_input)

        for param_name, param in model.named_parameters():
            if self.fold_bn:
                param = self._fold_batchnorm(model, param_name, param, self.named_modules)
            if not is_initialized:
                # Initialize the maskers
                masker = zeros_mask_dict[param_name]
                masker.use_double_copies = self.use_double_copies
                masker.mask_on_forward_only = self.mask_on_forward_only
                # register for the backward hook of the parameters
                if self.mask_gradients:
                    masker.backward_hook_handle = param.register_hook(masker.mask_gradient)

                self.is_initialized = True
                if not self.skip_first_minibatch:
                    self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)
            else:
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)

    def on_minibatch_begin(self, model, epoch, minibatch_id, minibatches_per_epoch,
                           zeros_mask_dict, meta, optimizer=None):
        set_masks = False
        global_mini_batch_id = epoch * minibatches_per_epoch + minibatch_id
        if ((minibatch_id > 0) and
            (self.mini_batch_pruning_frequency != 0) and
            (global_mini_batch_id % self.mini_batch_pruning_frequency == 0)):
            # This is _not_ the first mini-batch of a new epoch (performed in on_epoch_begin)
            # and a pruning step is scheduled
            set_masks = True

        if self.skip_first_minibatch and global_mini_batch_id == 1:
            # Because we skipped the first mini-batch of the first epoch (global_mini_batch_id == 0)
            set_masks = True

        for param_name, param in model.named_parameters():
            if set_masks:
                if self.fold_bn:
                    param = self._fold_batchnorm(model, param_name, param, self.named_modules)
                self.pruner.set_param_mask(param, param_name, zeros_mask_dict, meta)
            zeros_mask_dict[param_name].apply_mask(param)

    def before_parameter_optimization(self, model, epoch, minibatch_id, minibatches_per_epoch,
                                      zeros_mask_dict, meta, optimizer):
        for param_name, param in model.named_parameters():
            zeros_mask_dict[param_name].revert_weights(param)

    def on_minibatch_end(self, model, epoch, minibatch_id, minibatches_per_epoch, zeros_mask_dict, optimizer):
        if self.discard_masks_at_minibatch_end:
            for param_name, param in model.named_parameters():
                zeros_mask_dict[param_name].mask = None

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        """The current epoch has ended"""
        if self.is_last_epoch:
            for param_name, param in model.named_parameters():
                masker = zeros_mask_dict[param_name]
                if self.keep_mask:
                    masker.use_double_copies = False
                    masker.mask_on_forward_only = False
                    masker.mask_tensor(param)
                if masker.backward_hook_handle is not None:
                    masker.backward_hook_handle.remove()
                    masker.backward_hook_handle = None


class LRPolicy(ScheduledTrainingPolicy):
    """Learning-rate decay scheduling policy.
    """
    def __init__(self, lr_scheduler):
        super(LRPolicy, self).__init__()
        self.lr_scheduler = lr_scheduler

    def on_epoch_end(self, model, zeros_mask_dict, meta, **kwargs):
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # Note: ReduceLROnPlateau doesn't inherit from _LRScheduler
            self.lr_scheduler.step(kwargs['metrics'][self.lr_scheduler.mode],
                                   epoch=meta['current_epoch'] + 1)
        else:
            self.lr_scheduler.step(epoch=meta['current_epoch'] + 1)
