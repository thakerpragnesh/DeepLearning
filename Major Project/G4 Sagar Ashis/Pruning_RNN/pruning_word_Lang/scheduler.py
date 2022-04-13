
import contextlib
import logging
import torch
from policy import PolicyLoss, LossComponent
from utils import model_device, normalize_module_name


__all__ = ["CompressionScheduler", "ParameterMasker", "create_model_masks_dict"]
msglogger = logging.getLogger()


class CompressionScheduler(object):
    def __init__(self, model, zeros_mask_dict=None, device=torch.device("cuda")):
        self.model = model
        self.device = device
        self.policies = {}
        self.sched_metadata = {}
        self.zeros_mask_dict = zeros_mask_dict or create_model_masks_dict(model)

    def add_policy(self, policy, epochs=None, starting_epoch=None, ending_epoch=None, frequency=1):
        """Add a new policy to the schedule."""
        assert (epochs is None and None not in (starting_epoch, ending_epoch, frequency)) or\
               (epochs is not None and all (c is None for c in (starting_epoch, ending_epoch)))

        if epochs is None:
            assert 0 <= starting_epoch < ending_epoch
            assert 0 < frequency <= (ending_epoch - starting_epoch)
            epochs = list(range(starting_epoch, ending_epoch, frequency))
        else:
            starting_epoch = epochs[0]
            ending_epoch = epochs[-1] + 1
            frequency = None

        for epoch in epochs:
            if epoch not in self.policies:
                self.policies[epoch] = [policy]
            else:
                self.policies[epoch].append(policy)
            assert len(self.policies[epoch]) > 0

        self.sched_metadata[policy] = {'starting_epoch': starting_epoch,
                                       'ending_epoch': ending_epoch,
                                       'frequency': frequency}

    def on_epoch_begin(self, epoch, optimizer=None, **kwargs):
        for policy in self.policies.get(epoch, list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            policy.on_epoch_begin(self.model, self.zeros_mask_dict, meta,
                                  **kwargs)

    def on_minibatch_begin(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                policy.on_minibatch_begin(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                          self.zeros_mask_dict, meta, optimizer)

    def before_backward_pass(self, epoch, minibatch_id, minibatches_per_epoch, loss, optimizer=None,
                             return_loss_components=False):
        # We pass the loss to the policies, which may override it
        overall_loss = loss
        loss_components = []
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy_loss = policy.before_backward_pass(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                          overall_loss, self.zeros_mask_dict)
                if policy_loss is not None:
                    curr_loss_components = self.verify_policy_loss(policy_loss)
                    overall_loss = policy_loss.overall_loss
                    loss_components += curr_loss_components

        if return_loss_components:
            return PolicyLoss(overall_loss, loss_components)

        return overall_loss

    def before_parameter_optimization(self, epoch, minibatch_id, minibatches_per_epoch, optimizer):
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                meta = self.sched_metadata[policy]
                meta['current_epoch'] = epoch
                policy.before_parameter_optimization(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                                     self.zeros_mask_dict, meta, optimizer)

    def on_minibatch_end(self, epoch, minibatch_id, minibatches_per_epoch, optimizer=None):
        self.mask_all_weights(is_forward=False)
        if epoch in self.policies:
            for policy in self.policies[epoch]:
                policy.on_minibatch_end(self.model, epoch, minibatch_id, minibatches_per_epoch,
                                        self.zeros_mask_dict, optimizer)

    def on_epoch_end(self, epoch, optimizer=None, **kwargs):
        for policy in self.policies.get(epoch, list()):
            meta = self.sched_metadata[policy]
            meta['current_epoch'] = epoch
            meta['optimizer'] = optimizer
            policy.on_epoch_end(self.model, self.zeros_mask_dict, meta,
                                **kwargs)

    def mask_all_weights(self, is_forward=True):
        for name, param in self.model.named_parameters():
            try:
                masker = self.zeros_mask_dict[name]
                if is_forward or not masker.mask_on_forward_only:
                    # When we mask on forward-pass only, we allow the gradients to change
                    # the weights.
                    masker.mask_tensor(param)
            except KeyError:
                name_parts = name.split('.')
                prefixed = name_parts[-1].startswith(FP_BKP_PREFIX)
                wrapped = name_parts[-2] == 'wrapped_module'
                if prefixed or wrapped:
                    if prefixed:
                        name_parts[-1] = name_parts[-1].replace(FP_BKP_PREFIX, '', 1)
                    if wrapped:
                        name_parts.pop(-2)
                    name = '.'.join(name_parts)
                    self.zeros_mask_dict[name].apply_mask(param)

    def state_dict(self):
        masks = {}
        for name, masker in self.zeros_mask_dict.items():
            masks[name] = masker.mask
        state = {'masks_dict': masks}
        return state

    def load_state_dict(self, state, normalize_dataparallel_keys=False):
        try:
            loaded_masks = state['masks_dict']
        except KeyError as exception:
            msglogger.error('could not load the CompressionScheduler state.'
                            ' masks_dict is missing from state')
            with contextlib.suppress(TypeError):
                msglogger.debug('Scheduler state keys are: {}'.format(', '.join(state)))
            raise

        if normalize_dataparallel_keys:
            loaded_masks = {normalize_module_name(k): v for k, v in loaded_masks.items()}
        device = model_device(self.model)
        for name, mask in self.zeros_mask_dict.items():
            masker = self.zeros_mask_dict[name]
            masker.mask = loaded_masks[name]
            if masker.mask is not None:
                masker.mask = masker.mask.to(device)

    def init_from_masks_dict(self, masks_dict, normalize_dataparallel_keys=False):
        for name, mask in self.zeros_mask_dict.items():
            if name not in masks_dict:
                masks_dict[name] = None
        state = {'masks_dict': masks_dict}
        self.load_state_dict(state, normalize_dataparallel_keys)

    @staticmethod
    def verify_policy_loss(policy_loss):
        if not isinstance(policy_loss, PolicyLoss):
            raise TypeError("A Policy's before_backward_pass must return either None or an instance of " +
                            PolicyLoss.__name__)
        curr_loss_components = policy_loss.loss_components
        if not isinstance(curr_loss_components, list):
            curr_loss_components = [curr_loss_components]
        if not all(isinstance(lc, LossComponent) for lc in curr_loss_components):
            raise TypeError("Expected an instance of " + LossComponent.__name__ +
                            " or a list of such instances")
        return curr_loss_components


class ParameterMasker(object):
    def __init__(self, param_name):
        self.mask = None                
        self.param_name = param_name    
        self.is_regularization_mask = False
        self.use_double_copies = False
        self.mask_on_forward_only = False
        self.unmasked_copy = None
        self.backward_hook_handle = None

    def apply_mask(self, parameter):
        if self.mask is None:
            return
        if self.use_double_copies:
            self.unmasked_copy = parameter.clone().detach()
        self.mask_tensor(parameter)
        if self.is_regularization_mask:
            self.mask = None
        return parameter

    def mask_tensor(self, tensor):
        if self.mask is not None:
            tensor.data.mul_(self.mask)

    def mask_gradient(self, gradient):
        if self.mask is not None:
            return gradient.mul(self.mask)

    def revert_weights(self, parameter):
        if not self.use_double_copies or self.unmasked_copy is None:
            # This parameter does not maintain double copies (this is OK)
            return
        parameter.data.copy_(self.unmasked_copy)
        self.unmasked_copy = None


def create_model_masks_dict(model):
    """A convenience function to create a dictionary of parameter maskers for a model"""
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return zeros_mask_dict
