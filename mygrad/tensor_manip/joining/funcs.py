from .ops import Concatenate, Dstack, Hstack, Stack, Vstack
from mygrad.tensor_base import Tensor

__all__ = ["concatenate",
           "stack",
           "dstack",
           "hstack",
           "vstack"]


def concatenate(*variables, axis=0):
    """ Joins multiple Tensors along an existing axis.

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        axis : Optional[int] (default=0)
            The index of the axis in the resulting Tensor's dimensions along which
            the input variables will be concatenated.

        Returns
        -------
        mygrad.Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([[1, 2], [3, 4]]) # shape=(2,2)
        >>> b = mg.Tensor([[5, 6], [7, 8]]) # shape=(2,2)
        >>> mg.concatenate(a, b, axis=1) #shape=(2,4)
        Tensor([[1 2 5 6]
                [3 4 7 8]])

        >>> c = mg.Tensor([[[1], [2], [3]]]) # shape=(1,3,1)
        >>> d = mg.Tensor([[[4], [5], [6]], [[7], [8], [9]]]) # shape=(2,3,1)
        >>> mg.concatenate(c, d) # shape=(3,3,1)
        Tensor([[[1]
                 [2]
                 [3]]

                [[4]
                 [5]
                 [6]]

                [[7]
                 [8]
                 [9]]])
        """
    return Tensor._op(Concatenate, *variables, op_kwargs=dict(axis=axis))


def dstack(*variables):
    """ Stacks multiple Tensors depth-wise, along the 3rd axis.
        0-D, 1-D, and 2-D Tensors are first reshaped to (1,1,1), (1,N,1),
        and (N,M,1) Tensors, respectively.

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        Returns
        -------
        mygrad.Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([[1, 2], [3, 4]]) # shape=(2,2)
        >>> b = mg.Tensor([[[-1, -2], [-3, -4]], [[-5, -6], [-7, -8]]]) # shape=(2,2,2)
        >>> mg.dstack(a, b) # shape=(2,2,3)
        Tensor([[[ 1 -1 -2]
                 [ 2 -3 -4]]

                [[ 3 -5 -6]
                 [ 4 -7 -8]]])
        """
    return Tensor._op(Dstack, *variables)


def hstack(*variables):
    """ Stacks multiple Tensors horizontally, along the 2nd (column) axis.
        1-D Tensors are joined along the 1st axis. 0-D Tensors are first
        reshaped to (1,) Tensors.

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        Returns
        -------
        mygrad.Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([[1], [2], [3]]) # shape=(3,1)
        >>> b = mg.Tensor([[4, 5], [6, 7], [8, 9]]) # shape=(3,2)
        >>> mg.hstack(a, b) # shape=(3,3)
        Tensor([[1 4 5]
                [2 6 7]
                [3 8 9]])"""
    return Tensor._op(Hstack, *variables)


def stack(*variables, axis=0):
    """ Stacks multiple Tensors along a new axis.

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        axis : Optional[int] (default=0)
            The index of the new axis in the resulting Tensor's dimensions. The
            input variables are stacked along this axis in the resulting Tensor.

        Returns
        -------
        mygrad.Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([1, 2, 3]) # shape=(3,)
        >>> b = mg.Tensor([4, 5, 6]) # shape=(3,)
        >>> mg.stack(a, b, axis=1) # shape=(3,2)
        Tensor([[1 4]
                [2 5]
                [3 6]])

        >>> c = mg.Tensor([[1, 2], [3, 4]]) # shape=(2,2)
        >>> d = mg.Tensor([[5, 6], [7, 8]]) # shape=(2,2)
        >>> mg.stack(c.data, d.data, axis=2) # shape=(2,2,2)
        Tensor([[[1 5]
                 [2 6]]

                [[3 7]
                 [4 8]]])"""
    return Tensor._op(Stack, *variables, op_kwargs=dict(axis=axis))


def vstack(*variables):
    """ Stacks multiple Tensors vertically, along the 1st (row) axis.
        0-D and 1-D Tensors are first reshaped into (1,1) and (1,N)
        Tensors, respectively.

        Parameters
        ----------
        variables : Sequence[mygrad.Tensor, numpy.ndarray]

        Returns
        -------
        mygrad.Tensor

        Examples
        --------
        >>> import mygrad as mg
        >>> a = mg.Tensor([1, 2, 3]) # shape=(3,)
        >>> b = mg.Tensor([4, 5, 6]) # shape=(3,)
        >>> mg.vstack(a, b) # shape=(2,3)
        Tensor([[1 2 3]
                [4 5 6]])"""
    return Tensor._op(Vstack, *variables)
