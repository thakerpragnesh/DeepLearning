
import argparse
from collections import OrderedDict
import contextlib
from copy import deepcopy
import logging
import operator
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import inspect
import norms

msglogger = logging.getLogger()


def model_device(model):
    if isinstance(model, nn.DataParallel):
        return model.src_device_obj
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        # Model has no parameters
        pass
    return 'cpu'


def optimizer_device_name(opt):
    return str(list(list(opt.state)[0])[0].device)


def to_np(var):
    return var.data.cpu().numpy()


def size2str(torch_size):
    if isinstance(torch_size, torch.Size):
        return size_to_str(torch_size)
    if isinstance(torch_size, (torch.FloatTensor, torch.cuda.FloatTensor)):
        return size_to_str(torch_size.size())
    if isinstance(torch_size, torch.autograd.Variable):
        return size_to_str(torch_size.data.size())
    if isinstance(torch_size, tuple) or isinstance(torch_size, list):
        return size_to_str(torch_size)
    raise TypeError


def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '('+(', ').join(['%d' % v for v in torch_size])+')'


def pretty_int(i):
    return "{:,}".format(i)


class MutableNamedTuple(dict):
    def __init__(self, init_dict):
        for k, v in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def assign_layer_fq_names(container, name=None):
    for name, module in container.named_modules():
        module.distiller_name = name


def find_module_by_fq_name(model, fq_mod_name):
    for module in model.modules():
        if hasattr(module, 'distiller_name') and fq_mod_name == module.distiller_name:
            return module
    return None


def normalize_module_name(layer_name):
    modules = layer_name.split('.')
    try:
        idx = modules.index('module')
    except ValueError:
        return layer_name
    del modules[idx]
    return '.'.join(modules)


def denormalize_module_name(parallel_model, normalized_name):
    fully_qualified_name = [mod_name for mod_name, _ in parallel_model.named_modules() if
                            normalize_module_name(mod_name) == normalized_name]
    if len(fully_qualified_name) > 0:
        return fully_qualified_name[-1]
    else:
        return normalized_name   # Did not find a module with the name <normalized_name>


def volume(tensor):
    if isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.cuda.FloatTensor):
        return np.prod(tensor.shape)
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        return np.prod(tensor)
    raise ValueError


def density(tensor):
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)


def sparsity(tensor):
    return 1.0 - density(tensor)


def sparsity_3D(tensor):
    if tensor.dim() != 4:
        return 0
    l1_norms = norms.filters_lp_norm(tensor, p=1, length_normalized=False)
    num_nonzero_filters = len(torch.nonzero(l1_norms))
    num_filters = tensor.size(0)
    return 1 - num_nonzero_filters / num_filters


def density_3D(tensor):
    return 1 - sparsity_3D(tensor)


def sparsity_2D(tensor):
    if tensor.dim() == 4:
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        view_2d = tensor
    else:
        return 0

    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs/num_structs


def density_2D(tensor):
    return 1 - sparsity_2D(tensor)





def sparsity_ch(tensor):
    if tensor.dim() != 4:
        return 0
    nonzero_channels = len(non_zero_channels(tensor))
    n_channels = tensor.size(1)
    return 1 - nonzero_channels/n_channels

def density_ch(tensor):
    """Channel-wise density for 4D tensors"""
    return 1 - sparsity_ch(tensor)


def sparsity_blocks(tensor, block_shape):
    if tensor.dim() != 4:
        raise ValueError("sparsity_blocks is only supported for 4-D tensors")

    if len(block_shape) != 4:
        raise ValueError("Block shape must be specified as a 4-element tuple")
    block_repetitions, block_depth, block_height, block_width = block_shape
    if not block_width == block_height == 1:
        raise ValueError("Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1")

    super_block_volume = volume(block_shape)
    num_super_blocks = volume(tensor) / super_block_volume

    num_filters, num_channels = tensor.size(0), tensor.size(1)
    kernel_size = tensor.size(2) * tensor.size(3)

    # Create a view where each block is a column
    if block_depth > 1:
        view_dims = (
            num_filters*num_channels//(block_repetitions*block_depth),
            block_repetitions*block_depth,
            kernel_size,
            )
    else:
        view_dims = (
            num_filters // block_repetitions,
            block_repetitions,
            -1,
            )
    view1 = tensor.view(*view_dims)

    # Next, compute the sums of each column (block)
    block_sums = view1.abs().sum(dim=1)
    nonzero_blocks = len(torch.nonzero(block_sums))
    return 1 - nonzero_blocks/num_super_blocks


def sparsity_matrix(tensor, dim):
    """Generic sparsity computation for 2D matrices"""
    if tensor.dim() != 2:
        return 0

    num_structs = tensor.size()[dim]
    nonzero_structs = len(torch.nonzero(tensor.abs().sum(dim=1-dim)))
    return 1 - nonzero_structs/num_structs


def sparsity_cols(tensor, transposed=True):
    if transposed:
        return sparsity_matrix(tensor, 0)
    return sparsity_matrix(tensor, 1)


def density_cols(tensor, transposed=True):
    return 1 - sparsity_cols(tensor, transposed)


def sparsity_rows(tensor, transposed=True):
    if transposed:
        return sparsity_matrix(tensor, 1)
    return sparsity_matrix(tensor, 0)


def density_rows(tensor, transposed=True):
    return 1 - sparsity_rows(tensor, transposed)


def model_sparsity(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    sparsity, _, _ = model_params_stats(model, param_dims, param_types)
    return sparsity


def model_params_size(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    _, _, sparse_params_cnt = model_params_stats(model, param_dims, param_types)
    return sparse_params_cnt


def model_params_stats(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    params_cnt = 0
    params_nnz_cnt = 0
    for name, param in model.state_dict().items():
        if param.dim() in param_dims and any(type in name for type in param_types):
            _density = density(param)
            params_cnt += torch.numel(param)
            params_nnz_cnt += param.numel() * _density
    model_sparsity = (1 - params_nnz_cnt/params_cnt)*100
    return model_sparsity, params_cnt, params_nnz_cnt





def model_numel(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    total_numel = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in param_types):
            total_numel += torch.numel(param)
    return total_numel


def activation_channels_l1(activation):
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_norms = view_2d.norm(p=1, dim=1)  # (batch*channels) x 1
        featuremap_norms_mat = featuremap_norms.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_norms_mat = activation.norm(p=1, dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_l1: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_norms_mat.mean(dim=0).cpu()


def activation_channels_means(activation):
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_means = view_2d.mean(dim=1)  # (batch*channels) x 1
        featuremap_means_mat = featuremap_means.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_means_mat = activation.mean(dim=1)  # batch x 1
    else:
        raise ValueError("activation_channels_means: Unsupported shape: ".format(activation.shape))
    # We need to move the results back to the CPU
    return featuremap_means_mat.mean(dim=0).cpu()


def activation_channels_apoz(activation):
    if activation.dim() == 4:
        view_2d = activation.view(-1, activation.size(2) * activation.size(3))  # (batch*channels) x (h*w)
        featuremap_apoz = view_2d.abs().gt(0).sum(dim=1).float() / (activation.size(2) * activation.size(3))  # (batch*channels) x 1
        featuremap_apoz_mat = featuremap_apoz.view(activation.size(0), activation.size(1))  # batch x channels
    elif activation.dim() == 2:
        featuremap_apoz_mat = activation.abs().gt(0).sum(dim=1).float() / activation.size(1)  # batch x 1
    else:
        raise ValueError("activation_channels_apoz: Unsupported shape: ".format(activation.shape))
    return 100 - featuremap_apoz_mat.mean(dim=0).mul(100).cpu()
@contextlib.contextmanager
def get_nonparallel_clone_model(model):
    clone_model = make_non_parallel_copy(model)
    try:
        yield clone_model
    finally:
        del clone_model


def set_seed(seed):
    """Seed the PRNG for the CPU, Cuda, numpy and Python"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_deterministic(seed=0):
    msglogger.debug('set_deterministic was invoked')
    if seed is None:
        seed = 0
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def yaml_ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def yaml_ordered_save(fname, ordered_dict):
    def ordered_dict_representer(self, value):
        return self.represent_mapping('tag:yaml.org,2002:map', value.items())

    yaml.add_representer(OrderedDict, ordered_dict_representer)

    with open(fname, 'w') as f:
        yaml.dump(ordered_dict, f, default_flow_style=False)


def float_range_argparse_checker(min_val=0., max_val=1., exc_min=False, exc_max=False):
    def checker(val_str):
        val = float(val_str)
        min_op, min_op_str = (operator.gt, '>') if exc_min else (operator.ge, '>=')
        max_op, max_op_str = (operator.lt, '<') if exc_max else (operator.le, '<=')
        if min_op(val, min_val) and max_op(val, max_val):
            return val
        raise argparse.ArgumentTypeError(
            'Value must be {} {} and {} {} (received {})'.format(min_op_str, min_val, max_op_str, max_val, val))
    if min_val >= max_val:
        raise ValueError('min_val must be less than max_val')
    return checker


def filter_kwargs(dict_to_filter, function_to_call):
    sig = inspect.signature(function_to_call)
    filter_keys = [param.name for param in sig.parameters.values() if (param.kind == param.POSITIONAL_OR_KEYWORD)]
    valid_args = {}
    invalid_args = {}

    for key in dict_to_filter:
        if key in filter_keys:
            valid_args[key] = dict_to_filter[key]
        else:
            invalid_args[key] = dict_to_filter[key]
    return valid_args, invalid_args


def convert_tensors_recursively_to(val, *args, **kwargs):
    """ Applies `.to(*args, **kwargs)` to each tensor inside val tree. Other values remain the same."""
    if isinstance(val, torch.Tensor):
        return val.to(*args, **kwargs)

    if isinstance(val, (tuple, list)):
        return type(val)(convert_tensors_recursively_to(item, *args, **kwargs) for item in val)

    return val


def model_setattr(model, attr_name, val, register=False):
    def split_name(name):
        if '.' in name:
            return name.rsplit('.', 1)
        else:
            return '', name
    modules_dict = OrderedDict(model.named_modules())
    lowest_depth_container_name, lowest_depth_attr_name = split_name(attr_name)
    while lowest_depth_container_name and lowest_depth_container_name not in modules_dict:
        container_name, attr = split_name(lowest_depth_container_name)
        lowest_depth_container_name = container_name
        lowest_depth_attr_name = '%s%s' % (attr, lowest_depth_attr_name)
    lowest_depth_container = modules_dict[lowest_depth_container_name]  # type: nn.Module

    if register and torch.is_tensor(val):
        if isinstance(val, nn.Parameter):
            lowest_depth_container.register_parameter(lowest_depth_attr_name, val)
        else:
            lowest_depth_container.register_buffer(lowest_depth_attr_name, val)
    else:
        setattr(lowest_depth_container, lowest_depth_attr_name, val)


def param_name_2_module_name(param_name):
    return '.'.join(param_name.split('.')[:-1])


def is_scalar(val):
    result = isinstance(val, torch.Tensor) and val.dim() == 0
    result |= np.isscalar(val)
    return result

def size_to_str(torch_size):
    """Convert a pytorch Size object to a string"""
    assert isinstance(torch_size, torch.Size) or isinstance(torch_size, tuple) or isinstance(torch_size, list)
    return '('+(', ').join(['%d' % v for v in torch_size])+')'

def density(tensor):
    nonzero = tensor.abs().gt(0).sum()
    return float(nonzero.item()) / torch.numel(tensor)

def sparsity(tensor):
    return 1.0 - density(tensor)




def density_3D(tensor):
    """Filter-wise density for 4D tensors"""
    return 1 - sparsity_3D(tensor)


def sparsity_2D(tensor):
    if tensor.dim() == 4:
        # For 4D weights, 2D structures are channels (filter kernels)
        view_2d = tensor.view(-1, tensor.size(2) * tensor.size(3))
    elif tensor.dim() == 2:
        # For 2D weights, 2D structures are either columns or rows.
        # At the moment, we only support row structures
        view_2d = tensor
    else:
        return 0

    num_structs = view_2d.size()[0]
    nonzero_structs = len(torch.nonzero(view_2d.abs().sum(dim=1)))
    return 1 - nonzero_structs/num_structs


def density_2D(tensor):
    """Kernel-wise sparsity for 4D tensors"""
    return 1 - sparsity_2D(tensor)




def sparsity_ch(tensor):
    """Channel-wise sparsity for 4D tensors"""
    if tensor.dim() != 4:
        return 0
    nonzero_channels = len(non_zero_channels(tensor))
    n_channels = tensor.size(1)
    return 1 - nonzero_channels/n_channels


def density_ch(tensor):
    """Channel-wise density for 4D tensors"""
    return 1 - sparsity_ch(tensor)


def sparsity_blocks(tensor, block_shape):
    """Block-wise sparsity for 4D tensors
    Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1
    """
    if tensor.dim() != 4:
        raise ValueError("sparsity_blocks is only supported for 4-D tensors")

    if len(block_shape) != 4:
        raise ValueError("Block shape must be specified as a 4-element tuple")
    block_repetitions, block_depth, block_height, block_width = block_shape
    if not block_width == block_height == 1:
        raise ValueError("Currently the only supported block shape is: block_repetitions x block_depth x 1 x 1")

    super_block_volume = volume(block_shape)
    num_super_blocks = volume(tensor) / super_block_volume

    num_filters, num_channels = tensor.size(0), tensor.size(1)
    kernel_size = tensor.size(2) * tensor.size(3)

    # Create a view where each block is a column
    if block_depth > 1:
        view_dims = (
            num_filters*num_channels//(block_repetitions*block_depth),
            block_repetitions*block_depth,
            kernel_size,
            )
    else:
        view_dims = (
            num_filters // block_repetitions,
            block_repetitions,
            -1,
            )
    view1 = tensor.view(*view_dims)

    # Next, compute the sums of each column (block)
    block_sums = view1.abs().sum(dim=1)
    nonzero_blocks = len(torch.nonzero(block_sums))
    return 1 - nonzero_blocks/num_super_blocks


def sparsity_matrix(tensor, dim):
    """Generic sparsity computation for 2D matrices"""
    if tensor.dim() != 2:
        return 0

    num_structs = tensor.size()[dim]
    nonzero_structs = len(torch.nonzero(tensor.abs().sum(dim=1-dim)))
    return 1 - nonzero_structs/num_structs


def sparsity_cols(tensor, transposed=True):
    if transposed:
        return sparsity_matrix(tensor, 0)
    return sparsity_matrix(tensor, 1)


def density_cols(tensor, transposed=True):
    """Column-wise density for 2D tensors"""
    return 1 - sparsity_cols(tensor, transposed)


def sparsity_rows(tensor, transposed=True):
    if transposed:
        return sparsity_matrix(tensor, 1)
    return sparsity_matrix(tensor, 0)


def density_rows(tensor, transposed=True):
    """Row-wise density for 2D tensors"""
    return 1 - sparsity_rows(tensor, transposed)


def model_sparsity(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Returns the model sparsity as a fraction in [0..1]"""
    sparsity, _, _ = model_params_stats(model, param_dims, param_types)
    return sparsity


def model_params_size(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Returns the size of the model parameters, w/o counting zero coefficients"""
    _, _, sparse_params_cnt = model_params_stats(model, param_dims, param_types)
    return sparse_params_cnt


def model_params_stats(model, param_dims=[2, 4], param_types=['weight', 'bias']):

    params_cnt = 0
    params_nnz_cnt = 0
    for name, param in model.state_dict().items():
        if param.dim() in param_dims and any(type in name for type in param_types):
            _density = density(param)
            params_cnt += torch.numel(param)
            params_nnz_cnt += param.numel() * _density
    model_sparsity = (1 - params_nnz_cnt/params_cnt)*100
    return model_sparsity, params_cnt, params_nnz_cnt




def model_numel(model, param_dims=[2, 4], param_types=['weight', 'bias']):
    """Count the number elements in a model's parameter tensors"""
    total_numel = 0
    for name, param in model.state_dict().items():
        # Extract just the actual parameter's name, which in this context we treat as its "type"
        if param.dim() in param_dims and any(type in name for type in param_types):
            total_numel += torch.numel(param)
    return total_numel