from einops import rearrange
from functools import wraps
import inspect
import torch
from typing import List, Union, Literal

def rearrange_wrapper(obj: torch.nn.Module, input: str, hidden: str = "$ c h w", output: str = None, qux: str = "$", 
                      for_in: Union[List[Union[str, int]], Literal["all"], Literal["firstonly"]] = "all",
                      for_out: Union[List[int], Literal["all"]] = "all") -> torch.nn.Module:
    """
    A wrapper function to modify the `forward` method of a PyTorch module by dynamically rearranging tensor dimensions
    during the forward pass. The function allows flexible input-output dimension mappings, ensuring that tensors
    conform to specified patterns before and after being passed through the module.

    In most cases because modifying the tensor's view does not extra memory.

    ### Parameters:
    - **obj** (`torch.nn.Module`):  
      The PyTorch module whose `forward` method will be wrapped and modified.
    
    - **input** (`str`):  
      A space-separated string representing the dimension names of the input tensor (e.g., `"b c h w"`).
    
    - **hidden** (`str`, optional):  
      A space-separated string representing the intermediate "hidden" dimension names used during the forward pass.  
      Default is `"$ c h w"`.
      
    - **output** (`str`, optional):  
      A space-separated string representing the dimension names of the output tensor. If not provided, it defaults to the same as `input`.
    
    - **qux** (`str`, optional):  
      The dimension name in `hidden` that will combine dimensions not explicitly listed in `hidden`.  
      Default is `"$"`.
    
    - **for_in** (`Union[List[Union[str, int]], Literal["all"], Literal["firstonly"]]`, optional):  
      Specifies which input arguments to the `forward` method should be rearranged:  
      - `"firstonly"`: Rearrange only the first input argument (default).  
      - `"all"`: Rearrange all input arguments.  
      - `List[str | int]`: Explicit list of argument names or indices to rearrange.
    
    - **for_out** (`Union[List[int], Literal["all"]]`, optional):  
      Specifies which outputs from the `forward` method should be rearranged:  
      - `"all"`: Rearrange all outputs (default).  
      - `List[int]`: Explicit list of output indices to rearrange.

    ### Example Usage:
    ```python
    from torch import nn
    from einops import rearrange

    class SimpleModel(nn.Module):
        def forward(self, x):
            # Example forward pass
            return x.mean(dim=1, keepdim=True)

    model = SimpleModel()

    # Wrap the model with rearrange_wrapper
    model = rearrange_wrapper(
        obj=model,
        input="b c h w",
        output="b h w",
        hidden="$ c h w",
        qux="$",
        for_in="firstonly",
        for_out="all"
    )

    # Input tensor
    x = torch.randn(2, 3, 4, 4)

    # Forward pass with automatic rearrangement
    y = model(x)  # Input is rearranged from "b c h w" -> "$ c h w", and output is rearranged back to "b h w"
    print(y.shape)  # Output shape: (2, 4, 4)
    ```
    """

    input = input.strip().split()
    output = output.strip().split() if output is not None else input
    hidden = hidden.strip().split()
    qux = qux.strip()

    # Args checking
    def is_unique(lst):
         return len(lst) == len(set(lst))
    
    for name, lst in zip(["input", "output", "hidden"], [input, output, hidden]):
        if not is_unique(lst):
            raise ValueError(f"`{name}` must be unique, but got {lst}.")

    if qux not in hidden:
        raise ValueError(f"`combine_to` {qux} must in {hidden}.")

    for dim in hidden:
        if dim == qux:
            continue
        if dim not in input:
            raise ValueError(f"Expected `{dim}` in input {input}")
        if dim not in output:
            raise ValueError(f"Expected `{dim}` in output {output}")
    
    idx_qux = hidden.index(qux)
    combine = [dim for dim in input if dim not in hidden]
    hidden[idx_qux] = f"({' '.join(combine)})"

    in_pattern = " ".join(input)
    out_pattern = " ".join(output)
    hidden_pattern = " ".join(hidden)

    forward = obj.forward

    def output_filter(output, origin_shape):
        if not isinstance(output, (list, tuple)):
            output = rearrange(output, f"{hidden_pattern} -> {out_pattern}", **{dim: origin_shape[dim] for dim in combine})
        else:
            output_ = []
            for i, output_item in enumerate(output):
                if for_out == "all" or i in for_out:
                    output_item = rearrange(output_item, f"{hidden_pattern} -> {out_pattern}", **{dim: origin_shape[dim] for dim in combine})
                output_.append(output_item)
            if isinstance(output, tuple):
                output_ = tuple(output_)
            output = output_
        return output
    
    signature = inspect.signature(forward)
    parameters = signature.parameters
    
    # params_offset = 1 if inspect.ismethod(forward) else 0
    params_offset = 0
    params_mapper = {index: name for index, (name, param) in enumerate(parameters.items())}

    if for_in == "firstonly":
        for_in = [params_mapper[0 + params_offset]]
    elif for_in == "all":
        for_in = [params_mapper[i] for i in range(len(parameters))]
    else:
        for_in_ = []
        for key in for_in:
            if isinstance(key, int):
                if key > len(parameters) - params_offset:
                    raise ValueError(f"Index `{key}` out of range for parameters.")
                key = params_mapper.get(key + params_offset)
            else:
                if key not in parameters:
                    raise ValueError(f"Parameter `{key}` not found in the function signature.")
            for_in_.append(key)

        if len(for_in_) == 0:
            raise ValueError("No valid parameters provided in `for_in`.")
        for_in = for_in_

    def process_args_kwargs(*args, **kwargs):
        bound_arguments = {}
        for i, (name, param) in enumerate(parameters.items()):
            if i < len(args):
                bound_arguments[name] = args[i]
            elif name in kwargs:
                bound_arguments[name] = kwargs[name]
            elif param.default != inspect.Parameter.empty:
                bound_arguments[name] = param.default
            else:
                raise ValueError(f"Missing required argument: `{name}`")
        
        return bound_arguments
    
    @wraps(forward)
    def rearrange_aspect_variadic(*args, **kwargs):
        kwargs = process_args_kwargs(*args, **kwargs)
        origin_shape = None
        for argi, argv in kwargs.items():
            if argi in for_in:
                if argv is None:
                    kwargs[argi] = argv
                    continue
                if not isinstance(argv, torch.Tensor):
                    raise ValueError(f"Parameter `{argi}` must be a `torch.Tensor`, but got {type(argv)}.")
                origin_shape_ = dict(zip(input, argv.shape))
                if origin_shape is None:
                    origin_shape = origin_shape_
                argv = rearrange(argv, f"{in_pattern} -> {hidden_pattern}")
                kwargs[argi] = argv
        output = forward(**kwargs)
        output = output_filter(output, origin_shape)
        return output

    obj.forward = rearrange_aspect_variadic
    return obj


# Monkey patching
setattr(torch.nn.Module, "rearrange", rearrange_wrapper)