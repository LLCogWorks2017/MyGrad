from numbers import Number

import numpy as np

from mygrad.operation_base import BroadcastableOp, Operation

__all__ = ["GetItem", "SetItem"]


def _is_int_array_index(index):
    """ Returns True if `index` contains any array-like integer-valued sequences

        Parameters
        ----------
        index : Tuple[Any]

        Returns
        -------
        bool """
    return any(
        np.issubdtype(np.asarray(ind).dtype, np.int_) and np.asarray(ind).ndim
        for ind in index
    )


def _is_bool_array_index(index):
    """ Returns True if `index` solely contains a boolean-valued array

        Parameters
        ----------
        index : Tuple[Any]

        Returns
        -------
        bool """
    return len(index) == 1 and np.issubdtype(np.asarray(index[0]).dtype, np.bool_)


class GetItem(Operation):
    """ Defines the __getitem__ interface for a Tensor, supporting back-propagation

        Supports back-propagation through all valid numpy-indexing (basic, advanced, mixed, etc.)"""

    def __call__(self, a, index):
        """ ``a[index]``

        Parameters
        ----------
        a : mygrad.Tensor
            The tensor whose entries are being accessed.

        index : valid-array-index
            An n-dimensional index for specifying entries or subregions of `a`.
            All means of numpy-array indexing (basic, advanced, mixed, etc) are
            supported.

        Returns
        -------
        numpy.ndarray
            The array returned by the get-item operation"""
        self.variables = (a,)
        self.index = index if isinstance(index, tuple) else (index,)
        out = a.data[index]

        self._used_distinct_indices = (
            np.shares_memory(a.data, out)
            or isinstance(out, Number)
            or _is_bool_array_index(self.index)
        )
        return a.data[index]

    def backward_var(self, grad, index, **kwargs):
        a = self.variables[index]
        inds = self.index

        if self._used_distinct_indices or grad.size == 0:
            out = np.zeros_like(a.data)
            out[inds] += grad
        else:
            # used non-purely-boolean advanced indexing
            inds = list(inds)

            # tally of how many of each object appear in index
            num_bool_arr = 0
            num_int_arr = 0
            num_none = 0
            num_int = 0
            ell_ind = -1

            # used to determine whether all integer array indices are next to one another or not
            # and if not, where indices must be inserted for broadcast compatibility
            pos_int_arr = []

            # tally objects
            for j, ind in enumerate(inds):
                if isinstance(ind, slice):
                    continue
                elif isinstance(ind, int):
                    num_int += 1
                    continue
                elif ind is None:
                    num_none += 1
                    continue
                elif isinstance(ind, type(Ellipsis)):
                    ell_ind = j
                    continue

                ind_arr = np.asarray(ind)
                if np.issubdtype(ind_arr.dtype, np.bool_):
                    num_bool_arr += ind_arr.ndim - 1
                elif np.issubdtype(ind_arr.dtype, np.int_):
                    num_int_arr = (
                        ind_arr.ndim if ind_arr.ndim > num_int_arr else num_int_arr
                    )
                    pos_int_arr.append(j)

            if a.ndim == len(pos_int_arr) + num_int:
                # all dimensions indexed with integer arrays or integers
                out_ind = np.ravel_multi_index(inds, a.shape, mode="wrap")

            else:
                # expand ellipsis or add omitted trailing slices
                missing_dims = a.ndim - num_bool_arr - len(inds) + num_none
                if ell_ind != -1:
                    if missing_dims:
                        # ellipsis must be expanded into >1 slice
                        inds = (
                            inds[:ell_ind]
                            + [slice(None)] * (missing_dims + 1)
                            + inds[ell_ind + 1 :]
                        )
                        pos_int_arr = [
                            j + missing_dims if j > ell_ind else j
                            for j in pos_int_arr
                        ]
                    else:
                        # ellipsis acting as a single slice
                        inds[ell_ind] = slice(None)
                elif missing_dims:
                    inds += [slice(None)] * missing_dims

                # whether all integer arrays next to one another or not
                # determines how new dims added to int array indices
                if len(pos_int_arr):
                    contig_int_arr = bool(
                        len(pos_int_arr) == pos_int_arr[-1] - pos_int_arr[0] + 1
                    )
                    if not contig_int_arr:
                        diffs = [
                            i - j - 1
                            for i, j in zip(pos_int_arr[1:], pos_int_arr[:-1])
                        ]

                offset = 0
                out_ind = []

                for j, ind in enumerate(inds[::-1]):
                    if ind is None:
                        # any newaxis objects can be ignored due to later ravel
                        pass

                    elif isinstance(ind, int):
                        # integers will get broadcast out by ravel_multi_index
                        out_ind.append(ind)
                        offset += 1

                    elif isinstance(ind, slice):
                        # convert slice to arange for appropriate dimension
                        star = 0 if ind.start is None else ind.start
                        stop = (
                            a.shape[-len(out_ind) - 1]
                            if ind.stop is None
                            else ind.stop
                        )
                        step = 1 if ind.step is None else ind.step

                        out_ind.append(
                            np.arange(star, stop, step).reshape(
                                (-1,) + (1,) * offset
                            )
                        )
                        offset += 1

                    else:
                        ind = np.asarray(ind)

                        if np.issubdtype(ind.dtype, np.bool_):
                            # can simply find the Trues in a boolean array
                            non_zero_bools = np.nonzero(ind)
                            for dim_inds in non_zero_bools:
                                out_ind.append(
                                    dim_inds.reshape((-1,) + (1,) * offset)
                                )
                                offset += 1

                        else:
                            if contig_int_arr:
                                # integer arrays must already be broadcast compatible with
                                # one another, so simply add trailing 1-dims
                                # can ignore any leading 1-dims, as these will broadcast out
                                # as needed or be undone in the subsequent ravel
                                out_ind.append(
                                    ind.reshape(ind.shape + (1,) * offset)
                                )
                                if a.ndim - j - 1 == pos_int_arr[0]:
                                    offset += num_int_arr
                            else:
                                # add trailing indices for integer arrays
                                broadcast_shape = (1,) * (
                                    a.ndim
                                    - pos_int_arr[-1]
                                    - num_bool_arr
                                    + num_none
                                    - 1
                                    )

                                for i, d in zip(ind.shape[::-1], diffs[::-1]):
                                    # insert 1-dims between dimensions index with int arrays
                                    broadcast_shape += (i,) + (1,) * d

                                ind = ind.reshape(
                                    ind.shape[: -len(diffs)]
                                    + (1,) * (len(inds) - num_none + num_bool_arr)
                                    + broadcast_shape[::-1]
                                )
                                out_ind.append(ind)
                                offset += 1
                out_ind = np.ravel_multi_index(out_ind[::-1], a.shape, mode="wrap")

            out = np.bincount(
                out_ind.ravel(), weights=grad.ravel(), minlength=a.size
            ).reshape(a.shape)
        return out


def _arr(*shape):
    """ Construct an array of a specified consisting of values [0, _arr.size)
        filled in row-major order.

        Parameters
        ----------
        *shape : int

        Returns
        -------
        numpy.ndarray"""
    return np.arange(np.prod(shape)).reshape(shape)


class SetItem(BroadcastableOp):
    """ Defines the __setitem__ interface for a Tensor, supporting back-propagation through
        both the tensor being set and the tensor whose .

        Supports back-propagation through all valid numpy-indexing (basic, advanced, mixed, etc.),
        as well as """

    def __call__(self, a, b, index):
        """ a[index] = b

            Parameters
            ----------
            a : mygrad.Tensor
                The tensor whose entries are being set. A copy of the underlying
                data is made if `a` is a non-constant tensor.

            b : mygrad.Tensor
                `b` must be broadcast-compatible with `a[index]`

            index : valid-array-index
                An n-dimensional index for specifying entries or subregions of `a`.
                All means of numpy-array indexing (basic, advanced, mixed, etc) are
                supported.

            Notes
            -----
            Additional computational overhead is required for back-propagation when
            `index` contains any integer-valued arrays, to accommodate for the scenario
            in which a single element is set multiple times."""

        out = np.copy(a.data) if not a.constant else a.data
        self.variables = (a, b)
        self.index = index if isinstance(index, tuple) else (index,)
        out[index] = b.data
        return out

    def backward_var(self, grad, index, **kwargs):
        a, b = self.variables
        if index == 0:
            grad = np.copy(grad)
            grad[self.index] = 0
            return grad
        elif index == 1:
            grad_sel = np.asarray(grad[self.index])

            # Basic indexing and indexing with a single boolean-array is trivial. The
            # gradient into b can just be accessed by indexing into `grad`.
            # Indexing with integer-valued arrays can be problematic, as the same
            # item can be specified multiple for "setting"; here only the last set-item
            # for that element has an effect. For example:
            #     x[np.array([0, 0])] = np.array([2, 3])  # `3` gets set to x[0]; 2 has no effect
            # Thus only that corresponding element in `grad` (that corresponding to `3`)
            # should be propagated back into b. Thus we must check to see if any items are
            # being set redundantly, and mask out any elements in `grad` corresponding to
            # the elements in `b` that weren't actually set.
            if (
                not np.shares_memory(grad_sel, grad)
                and grad_sel.size > 0
                and grad_sel.ndim > 0
                and not _is_bool_array_index(self.index)
                and _is_int_array_index(self.index)
            ):
                # create an array of unique elements, and see if indexing into it produces
                # any redundant elements
                unique = _arr(*grad.shape)
                sub_sel = unique[self.index].flat
                elements, first_inds, = np.unique(
                    np.flip(sub_sel, axis=0), return_index=True
                )
                if len(first_inds) < len(sub_sel):
                    # one or more elements were set redundantly, identify the entries in `b`
                    # that actually were set to those elements (the last-most set-item calls
                    # for those elements) and propagate only the corresponding elements from grad

                    first_inds = (len(sub_sel) - 1) - first_inds
                    mask = np.zeros_like(sub_sel)
                    mask[first_inds] = 1
                    mask = mask.reshape(grad_sel.shape)
                    grad_sel *= mask

            # handle the edge case of "projecting down" on setitem. E.g:
            # x = Tensor([0, 1, 2])
            # y = Tensor([3])
            # x[0] = y  # this is legal since x[0] and y have the same size
            if grad_sel.ndim < b.ndim:
                if grad_sel.size == b.size:
                    grad_sel = grad_sel.reshape(b.shape)
                else:
                    # Broadcasting occurred during set-item and `b` contains
                    # excess leading singleton dimensions. Make `grad_sel`
                    # commensurate with `b` for subsequent `reduce_broadcast`
                    # to work
                    grad_sel = grad_sel[(np.newaxis,) * (b.ndim - grad_sel.ndim)]

            return grad_sel
        else:
            raise IndexError()  # pragma: no cover
