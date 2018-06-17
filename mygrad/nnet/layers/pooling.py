from mygrad.operation_base import Operation
from mygrad.tensor_base import Tensor
import numpy as np
from numbers import Integral
from mygrad.nnet.layers.utils import sliding_window_view


class MaxPoolND(Operation):
    scalar_only = True

    def __call__(self, x, pool, stride):
        """ Perform max-pooling over the last N dimensions of a data batch.

            Extended Summary
            ----------------
            The data consists of N trailing axes to be pooled over, denoted by `C0, ...`. These
            can be preceded, optionally, by un-pooled axes, denoted by `(N0, ...)`. The dimensions
            of the window over which pooling is performed is denoted by `P0, ...`. The window
            is placed with stride values `S0, ...`.

            Ultimately the pooled channels have a shape `C0', ...`.

            Parameters
            ----------
            x : mygrad.Tensor, shape=([(N0, ...), C0, ...])
                The data batch; to be pooled along the H and W axes

            pool : Tuple[Integral, ...], (P0, ...)
                The extent of the pooling window along the (C0, ...) axes, respectively. The
                length of `pool` determines N - the number of trailing dimensions to pool over.

            stride : Union[Integral, Tuple[Integral, ...]], (S0, ...)
                The spacing used to place the pooling window, along (P0, ...) axes, respectively.
                If a single value is provided, it is used for all N pooling axes.

            Returns
            -------
            numpy.ndarray, shape=((N0, ...), C0', ...)
                The pooled data batch.

            Notes
            -----
            Only 'valid' placements of the pooling window are permitted - the pooling
            window cannot extend passed the "boundaries" of the data
            dimensions.
            """
        self.variables = (x,)  # data: ((N0, ...), C0, ...)
        x = np.copy(x.data)  # prevent window-view weirdness with views

        pool = np.asarray(pool, dtype=int)
        assert all(i > 0 for i in pool)
        assert x.ndim >= len(pool), "The number of pooled dimensions cannot exceed the dimensionality of the data."
        
        stride = np.array([stride]*len(pool)) if isinstance(stride, Integral) else np.asarray(stride, dtype=int)
        assert len(stride) == len(pool) and all(s >= 1 and isinstance(s, Integral) for s in stride)

        self.pool = pool      # (P0, ...)
        self.stride = stride  # (S0, ...)

        num_pool = len(pool)
        num_no_pool = x.ndim - num_pool

        x_shape = np.array(x.shape[num_no_pool:])
        w_shape = pool

        out_shape = (x_shape - w_shape) / stride + 1

        if not all(i.is_integer() and i > 0 for i in out_shape):
            msg = "Stride and kernel dimensions are incompatible: \n"
            msg += "Input dimensions: {}\n".format(tuple(x_shape))
            msg += "Stride dimensions: {}\n".format(tuple(stride))
            msg += "Pooling dimensions: {}\n".format(tuple(w_shape))
            raise ValueError(msg)

        pool_axes = tuple(-(i + 1) for i in range(num_pool))
        
        # (G0, ...) is the tuple of grid-positions for placing each window (not including stride)
        # sliding_window_view(x): ((N0, ...), C0, ...)          -> (G0, ..., (N0, ...), P0, ...)
        # max-pool:               (G0, ..., (N0, ...), P0, ...) -> (G0, ..., (N0, ...))
        maxed = sliding_window_view(x, self.pool, self.stride).max(axis=pool_axes)
        axes = tuple(range(maxed.ndim))

        # (G0, ..., (N0, ...)) -> ((N0, ...), G0, ...)
        out = maxed.transpose(axes[-num_no_pool:] + axes[:-num_no_pool])
        return out if out.flags['C_CONTIGUOUS'] else np.ascontiguousarray(out)

    def backward_var(self, grad, index, **kwargs):
        """ Parameters
            ----------
            grad : numpy.ndarray, shape=((N0, ...), G0, ...),
            index : int"""
        var = self.variables[index]
        x = var.data
        num_pool = len(self.pool)

        sl = sliding_window_view(x, self.pool, self.stride)
        grid_shape = sl.shape
        maxed = sl.reshape(*sl.shape[:-num_pool], -1).argmax(-1)
        axes = tuple(range(maxed.ndim))

        # argmax within a given flat-window
        maxed = maxed.transpose(axes[num_pool:] + axes[:num_pool])  # ((N0, ...), G0, ...)

        # flat-index offset associated with reshaped window within `x`
        row_major_offset = tuple(np.cumprod(x.shape[-num_pool:][:0:-1])[::-1]) + (1,)

        # flat index of argmax, updated based on position within window, according to shape of `x`
        in_window_offset = sum(
            ind * off for ind, off in zip(np.unravel_index(maxed, self.pool),
                                          row_major_offset))

        # flat-index of strided window placement, relative to `x`
        window_offset = sum(ind * s * off for ind, s, off in zip(np.indices(grid_shape[:num_pool]),
                                                                 self.stride,
                                                                 row_major_offset))

        # indices required to traverse pool-axis-flattened array
        # ((N0, ...) G0*...)
        flat_grid_shape = (*maxed.shape[:-num_pool], np.prod(maxed.shape[-num_pool:]))
        index = np.indices(flat_grid_shape)

        # update trailing indices to traverse location of max entries within pooled axes
        index[-1] = (in_window_offset + window_offset).reshape(*flat_grid_shape[:-1], -1)

        # accumulate gradient within pool-axis-flattened dx, then reshape to match shape of `x`
        dx = np.zeros(x.shape[:-num_pool] + (np.prod(x.shape[-num_pool:]),))
        np.add.at(dx, tuple(index), grad.reshape(*x.shape[:-num_pool], -1))
        var.backward(dx.reshape(x.shape), **kwargs)


def max_pool(x, pool, stride, constant=False):
    """ Perform max-pooling over the last N dimensions of a data batch.

        Parameters
        ----------
        x : array_like, shape=([(N0, ...), C0, ...])
            The data batch; to be pooled along the H and W axes

        pool : Tuple[Integral, ...]
            (P0, ...)

            The extent of the pooling window along the (C0, ...) axes, respectively. The
            length of `pool` determines N - the number of trailing dimensions to pool over.

        stride : Union[Integral, Tuple[Integral, ...]]
            The spacing used to place the pooling window, along (P0, ...) axes, respectively.
            If a single value is provided, it is used for all N pooling axes.

        constant : bool, optional (default=False)
            If True, the resulting Tensor is a constant.

        Returns
        -------
        mygrad.Tensor, shape=((N0, ...), C0', ...)
            The pooled data batch.

        Examples
        --------

        >>> import numpy as np
        >>> from mygrad.nnet.layers import max_pool
        >>> from mygrad import Tensor
        >>> x = Tensor(np.random.rand(10, 3, 12, 12))
        >>> pool = (2, 2)   # 2x2 pooling over the last axes
        >>> stride = (2, 1) # Apply 2x1 stride
        >>> out = max_pool(x, pool, stride)  # max-pooled Tensor
        >>> out.shape
        (10, 3, 6, 11)

        Notes
        -----
        Only 'valid' placements of the pooling window are permitted - the pooling
        window cannot extend passed the "boundaries" of the data
        dimensions.
        """
    return Tensor._op(MaxPoolND, x, op_args=(pool, stride), constant=constant)
