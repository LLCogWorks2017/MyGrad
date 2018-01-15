from .multivar_operations import MultiVarOperation
import numpy as np
from itertools import accumulate, zip_longest


class Concatenate(MultiVarOperation):
    def __call__(self, *input_vars, axis=0):
        for i, dim in enumerate(list(zip_longest(*[var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if i == axis:
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.axis = axis
        self.indices = list(accumulate([var.data.shape[axis] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.concatenate([var.data for var in input_vars], axis=axis)

        return out

    def backward(self, grad):
        for index, var in enumerate(self.variables):
            if not var.constant:
                self.backward_var(grad, index)

    def backward_var(self, grad, index):
        var = self.variables[index]
        grad_slice = [slice(None, None, None) if dim is not self.axis else slice(self.indices[index], self.indices[index+1]) for dim in range(len(var.data.shape))]
        var.backward(grad[grad_slice])
