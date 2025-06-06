import warnings
import torch
from torch import nn
from typing import Callable
from collections import OrderedDict

class ExtraContext:
    """
    A context manager for managing extra losses, hooks, and objects in a PyTorch module.
    This context manager allows you to register additional losses, hooks, and objects
    that can be used during training or evaluation. It ensures that these additional
    components are properly cleaned up after use.
    
    Usage:
        with ExtraContext(model) as ctx:
            # Register extra losses, hooks, or objects
            model.extra_context.add_loss("loss_name", loss_tensor)
            model.extra_context.add_hook("hook_name", hook_function)
            model.extra_context.add_object("object_name", some_object)
    """


    def __init__(self, context: nn.Module, logger:Callable = None):
        self.__context = context
        self.logger = logger

        self.__registed_module = OrderedDict()
        self.__extra_losses = []
        self.__extra_hooks = []

        self.__extra_objects = {}

    def __enter__(self):
        for prefix, module in self.__context.named_modules():
            if module in self.__registed_module:
                warnings.warn(f"Submodule{prefix} is contained multiple times by different modules. This can lead to potential problems.", UserWarning)
                self.__registed_module[module].append(prefix)
            else:
                self.__registed_module[module] = [prefix,]
                if hasattr(module, "extra_context"):
                    raise ValueError(f"Name domain conflict. The module {prefix} already has `extra_context`.")
                setattr(module, "extra_context", self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__extra_losses = None
        self.__extra_hooks = None
        self.__extra_objects = None
        for module in self.__registed_module.keys():
            if not hasattr(module, "extra_context"):
                raise RuntimeError(
                    f"Module {module.__class__.__name__} has no context to remove."
                )
            delattr(module, "extra_context")
    
    def __getitem__(self, key):
        if self.__extra_objects is None:
            raise ValueError("Objects dict has been cleared. Users should not access context manager after exiting context. This could be a bug.")
        return self.__extra_objects[key]

    def __setitem__(self, key, value):
        if self.__extra_objects is None:
            raise ValueError("Objects dict has been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_objects[key] = value

    def add_loss(self, prefix:str, loss: torch.Tensor):
        """
        Add a loss term to the context.
        """
        if self.__extra_losses is None:
            raise ValueError("Losses have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_losses.append((prefix, loss))

    def add_hook(self, prefix:str, hook: Callable):
        """
        Add a hook to the context.
        """
        if self.__extra_hooks is None:
            raise ValueError("Hooks have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        self.__extra_hooks.append((prefix, hook))

    def log(self, *args, **kwargs):
        if self.logger is None:
            warnings.warn("No logger is set for ExtraContext. Logging will be ignored.", UserWarning, stacklevel=2)
            return
        res = self.logger(*args, **kwargs)
        return res
    
    @property
    def losses(self):
        if self.__extra_losses is None:
            raise ValueError("Losses have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        return self.__extra_losses

    @property
    def hooks(self):
        if self.__extra_hooks is None:
            raise ValueError("Hooks have been cleared. Users should not access context manager after exiting context. This could be a bug.")
        return self.__extra_hooks
        

def register_extra_loss(module: nn.Module, loss_term: torch.Tensor, prefix: str = None):
    """
    Register an extra loss term to a module.
    """
    if not hasattr(module, "extra_context"):
        warnings.warn(f"Training does not launch with an ExtraContext. This loss will be ignored.", UserWarning,stacklevel=2)
        return
    module.extra_context.add_loss(prefix, loss_term)

def register_extra_hook(module: nn.Module, hook: Callable, prefix: str = None):
    """
    Register an extra hook to a module.
    """
    if not hasattr(module, "extra_context"):
        warnings.warn(f"Training does not launch with an ExtraContext. This hook will be ignored.", UserWarning,stacklevel=2)
        return
    module.extra_context.add_hook(prefix, hook)

def get_extra_context(module: nn.Module):
    """
    Get the extra context of a module.
    """
    if not hasattr(module, "extra_context"):
        warnings.warn(f"Training does not launch with an ExtraContext. This will return None.", UserWarning, stacklevel=2)
        return None
    return module.extra_context

def log_extra(module: nn.Module, *args, **kwargs):
    """
    Log extra information to the module's logger.
    """
    if not hasattr(module, "extra_context"):
        warnings.warn(f"Training does not launch with an ExtraContext. Logging will be ignored.", UserWarning, stacklevel=2)
        return
    return module.extra_context.log(*args, **kwargs)