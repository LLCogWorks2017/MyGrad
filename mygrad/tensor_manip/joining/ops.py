from mygrad.operation_base import Operation
import numpy as np
from itertools import accumulate, zip_longest

__all__ = ["Concatenate",
           "Stack",
           "Dstack",
           "Hstack",
           "Vstack"]


class Concatenate(Operation):
    def __call__(self, *input_vars, axis=0):
        for i, dim in enumerate(list(zip_longest(*[(0,) if not var.data.ndim else var.data.shape for var in input_vars]))):
            if i == 0 and dim.count(0) != 0:
                assert False, "zero-dimensional Tensors cannot be concatenated"
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if i == axis:
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"
        assert axis < input_vars[0].ndim, "axis {} is out of bounds for Tensor of dimension {}".format(axis, input_vars[0].ndim)

        self.variables = input_vars
        self.axis = axis
        self.indices = list(accumulate([var.data.shape[axis] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.concatenate([var.data for var in input_vars], axis=axis)

        return out

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        grad_slice = [slice(None, None, None) if dim is not self.axis else slice(self.indices[index], self.indices[index+1]) for dim in range(var.data.ndim)]
        var.backward(grad[grad_slice])


class Stack(Operation):
    def __call__(self, *input_vars, axis=0):
        assert len({var.data.shape for var in input_vars}) == 1, "all input Tensors must have the same shape"
        assert axis < input_vars[0].ndim+1, "axis {} is out of bounds for Tensor of dimension {}".format(axis, input_vars[0].ndim+1)

        self.variables = input_vars
        self.axis = axis
        newaxis = [slice(None, None, None) if dim is not axis else None for dim in range(input_vars[0].data.ndim+1)]
        self.indices = list(accumulate([var.data[newaxis].shape[axis] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.stack([var.data for var in input_vars], axis=axis)

        return out

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        grad_slice = [slice(None, None, None) if dim is not self.axis else slice(self.indices[index], self.indices[index+1]) for dim in range(var.data.ndim+1)]
        var.backward(np.squeeze(grad[grad_slice], axis=self.axis))


class Dstack(Operation):
    def __call__(self, *input_vars):
        for i, dim in enumerate(list(zip_longest(*[var.data[np.newaxis,np.newaxis,np.newaxis].shape if not var.data.ndim else var.data[np.newaxis,:,np.newaxis].shape if var.data.ndim is 1 else var.data[...,np.newaxis].shape if var.data.ndim is 2 else var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if i == 2:
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.indices = list(accumulate([1 if (0,1,2).__contains__(var.data.ndim) else var.data.shape[2] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.dstack([var.data for var in input_vars])

        return out

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        if not var.data.ndim:
            var.backward(np.asscalar(grad[:,:,self.indices[index]:self.indices[index+1]]))
        elif var.data.ndim == 1:
            var.backward(np.squeeze(grad[:,:,self.indices[index]:self.indices[index+1]], axis=(0,2)))
        elif var.data.ndim == 2:
            var.backward(np.squeeze(grad[:,:,self.indices[index]:self.indices[index+1]], axis=2))
        else:
            var.backward(grad[:,:,self.indices[index]:self.indices[index+1]])


class Hstack(Operation):
    def __call__(self, *input_vars):
        for i, dim in enumerate(list(zip_longest(*[var.data[np.newaxis].shape if not var.data.ndim else var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if (i == 0 and (input_vars[0].data.ndim == 1 or not input_vars[0].data.ndim)) or (i == 1):
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.axis = 0 if input_vars[0].data.ndim is 1 else 0 if not input_vars[0].data.ndim else 1
        self.indices = list(accumulate([1 if not var.data.ndim else var.data.shape[0] if not self.axis else var.data.shape[1] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.hstack([var.data for var in input_vars])
        return out

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        if not var.data.ndim:
            var.backward(np.asscalar(grad[self.indices[index]:self.indices[index+1]]))
        elif not self.axis:
            var.backward(grad[self.indices[index]:self.indices[index+1]])
        else:
            var.backward(grad[:,self.indices[index]:self.indices[index+1]])


class Vstack(Operation):
    def __call__(self, *input_vars):
        for i, dim in enumerate(list(zip_longest(*[var.data[np.newaxis,np.newaxis].shape if not var.data.ndim else var.data[np.newaxis].shape if var.data.ndim is 1 else var.data.shape for var in input_vars]))):
            assert dim.count(None) == 0, "all input Tensors must have the same number of dimensions"

            if i == 0:
                pass
            else:
                assert dim.count(dim[0]) == len(dim), "all input Tensor dimensions except for the concatenation axis must match exactly"

        self.variables = input_vars
        self.indices = list(accumulate([1 if (0,1).__contains__(var.data.ndim)  else var.data.shape[0] for var in input_vars]))
        self.indices.insert(0,0)
        out = np.vstack([var.data for var in input_vars])

        return out

    def backward_var(self, grad, index, **kwargs):
        var = self.variables[index]
        if not var.data.ndim:
            var.backward(np.asscalar(grad[self.indices[index]:self.indices[index+1]]))
        elif var.data.ndim == 1:
            var.backward(np.squeeze(grad[self.indices[index]:self.indices[index+1]], axis=0))
        else:
            var.backward(grad[self.indices[index]:self.indices[index+1]])
