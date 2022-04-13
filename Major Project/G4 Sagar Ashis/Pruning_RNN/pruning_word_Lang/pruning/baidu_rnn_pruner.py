
import torch
def create_mask_threshold_criterion(tensor, threshold):
    """Create a tensor mask using a threshold criterion. """
    with torch.no_grad():
        mask = torch.gt(torch.abs(tensor), threshold).type(tensor.type())
        return mask

class BaiduRNNPruner(object):
    def __init__(self, name, q, ramp_epoch_offset, ramp_slope_mult, weights):
        # Initialize the pruner, using a configuration that originates from the
        # schedule YAML file.
        self.name = name
        self.params_names = weights
        assert self.params_names

        # This is the 'q' value that appears in equation (1) of the paper
        self.q = q
        # This is the number of epochs to wait after starting_epoch, before we
        # begin ramping up the pruning rate.
        # In other words, between epochs 'starting_epoch' and 'starting_epoch'+
        # self.ramp_epoch_offset the pruning slope is 'self.start_slope'.  After
        # that, the slope is 'self.ramp_slope'
        self.ramp_epoch_offset = ramp_epoch_offset
        self.ramp_slope_mult = ramp_slope_mult
        self.ramp_slope = None
        self.start_slope = None

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.params_names:
            return

        starting_epoch = meta['starting_epoch']
        current_epoch = meta['current_epoch']
        ending_epoch = meta['ending_epoch']
        freq = meta['frequency']

        ramp_epoch = self.ramp_epoch_offset + starting_epoch

        # Calculate start slope
        if self.start_slope is None:
            # We want to calculate these values only once, and then cache them.
            self.start_slope = (2 * self.q * freq) / (2*(ramp_epoch - starting_epoch) + 3*(ending_epoch - ramp_epoch))
            self.ramp_slope = self.start_slope * self.ramp_slope_mult

        if current_epoch < ramp_epoch:
            eps = self.start_slope * (current_epoch - starting_epoch + 1) / freq
        else:
            eps = (self.start_slope * (ramp_epoch - starting_epoch + 1) +
                   self.ramp_slope  * (current_epoch  - ramp_epoch + 1)) / freq

        # After computing the threshold, we can create the mask
        zeros_mask_dict[param_name].mask = create_mask_threshold_criterion(param, eps)
