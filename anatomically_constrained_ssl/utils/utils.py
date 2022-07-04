import copy
import warnings

import torch


def patch_module(module: torch.nn.Module, inplace: bool = True) -> torch.nn.Module:
    """Replace Relu layers in a model with LeakyRelu layers.

    Args:
        module (torch.nn.Module):
            The module in which you would like to replace Relu layers.
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Raises:
        UserWarning if no layer is modified.

    Returns:
        torch.nn.Module
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)
    changed = _patch_relu_layers(module)
    if not changed:
        warnings.warn("No layer was modified by patch_module!", UserWarning)
    return module


def _patch_relu_layers(module: torch.nn.Module) -> bool:
    """
    Recursively iterate over the children of a module and replace them if
    they are a Relu layer. This function operates in-place.

    Returns:
        Flag indicating if a layer was modified.
    """
    changed = False
    for name, child in module.named_children():
        if isinstance(child, torch.nn.ReLU):
            new_module = torch.nn.LeakyReLU(0.2)
        else:
            new_module = None

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed = changed or _patch_relu_layers(child)
    return changed
