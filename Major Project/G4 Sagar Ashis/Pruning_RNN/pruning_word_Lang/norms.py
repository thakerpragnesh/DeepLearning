
import torch
import numpy as np
from functools import partial
from random import uniform


__all__ = ["kernels_lp_norm", "channels_lp_norm", "filters_lp_norm",
           "kernels_norm", "channels_norm", "filters_norm", "sub_matrix_norm",
           "rows_lp_norm", "cols_lp_norm",
           "rows_norm", "cols_norm",
           "l1_norm", "l2_norm", "max_norm"]


class NamedFunction:
    def __init__(self, f, name):
        self.f = f
        self.name = name

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)

    def __str__(self):
        return self.name


def _max_norm(t, dim=1):
    maxv, _ = t.abs().max(dim=dim)
    return maxv


l1_norm = NamedFunction(partial(torch.norm, p=1, dim=1), "L1")
l2_norm = NamedFunction(partial(torch.norm, p=2, dim=1), "L2")
max_norm = NamedFunction(_max_norm, "Max")


def kernels_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return kernels_norm(param, norm_fn, group_len, length_normalized)


def kernels_norm(param, norm_fn, group_len=1, length_normalized=False):
    assert param.dim() == 4, "param has invalid dimensions"
    group_size = group_len * np.prod(param.shape[2:])
    return generic_norm(param.view(-1, group_size), norm_fn, group_size, length_normalized, dim=1)


def channels_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return channels_norm(param, norm_fn, group_len, length_normalized)


def channels_norm(param, norm_fn, group_len=1, length_normalized=False):
    assert param.dim() in (2, 4), "param has invalid dimensions"
    if param.dim() == 2:
        return cols_norm(param, norm_fn, group_len, length_normalized)
    param = param.transpose(0, 1).contiguous()
    group_size = group_len * np.prod(param.shape[1:])
    return generic_norm(param.view(-1, group_size), norm_fn, group_size, length_normalized, dim=1)


def filters_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return filters_norm(param, norm_fn, group_len, length_normalized)


def filters_norm(param, norm_fn, group_len=1, length_normalized=False):
    assert param.dim() == 4, "param has invalid dimensions"
    group_size = group_len * np.prod(param.shape[1:])
    return generic_norm(param.view(-1, group_size), norm_fn, group_size, length_normalized, dim=1)


def sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim):
    assert param.dim() == 2, "param has invalid dimensions"
    group_size = group_len * param.size(abs(dim - 1))
    return generic_norm(param, norm_fn, group_size, length_normalized, dim)


def rows_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=1)


def rows_norm(param, norm_fn, group_len=1, length_normalized=False):
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=1)


def cols_lp_norm(param, p=1, group_len=1, length_normalized=False):
    assert p in (1, 2)
    norm_fn = l1_norm if p == 1 else l2_norm
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=0)


def cols_norm(param, norm_fn, group_len=1, length_normalized=False):
    return sub_matrix_norm(param, norm_fn, group_len, length_normalized, dim=0)


def generic_norm(param, norm_fn, group_size, length_normalized, dim):
    with torch.no_grad():
        if dim is not None:
            norm = norm_fn(param, dim=dim)
        else:
            # The norm may have been specified as part of the norm function
            norm = norm_fn(param)
        if length_normalized:
            norm = norm / group_size
        return norm



def num_structs_to_prune(n_elems, group_len, fraction_to_prune, rounding_fn):
    n_structs_to_prune = rounding_fn(fraction_to_prune * n_elems)
    n_structs_to_prune = int(rounding_fn(n_structs_to_prune * 1. / group_len) * group_len)
    if n_structs_to_prune == n_elems and fraction_to_prune != 1.0:
        n_structs_to_prune = n_elems - group_len
    return n_structs_to_prune


def k_smallest_elems(mags, k, noise):
    mags *= e_greedy_normal_noise(mags, noise)
    k_smallest_elements, _ = torch.topk(mags, k, largest=False, sorted=True)
    return k_smallest_elements, mags

