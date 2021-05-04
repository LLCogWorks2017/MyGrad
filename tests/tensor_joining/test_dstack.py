from mygrad.tensor_base import Tensor
from mygrad import dstack

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_dstack_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))
    c = Tensor(a)
    b = b

    result = np.dstack((a, b))
    o = dstack(c, b)

    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


def test_dstack_backward():
    a = Tensor([3])
    b = Tensor(2)
    c = Tensor([[1]])
    d = Tensor([[[4, 5]]])
    e = dstack(a, b, c, d)
    e.backward([[[-3, -2, -1, -4, -5]]])

    assert a.data.shape == a.grad.shape
    assert np.allclose(a.data, np.abs(a.grad))

    assert b.data.shape == b.grad.shape
    assert np.allclose(b.data, np.abs(b.grad))

    assert c.data.shape == c.grad.shape
    assert np.allclose(c.data, np.abs(c.grad))

    assert d.data.shape == d.grad.shape
    assert np.allclose(d.data, np.abs(d.grad))


    e = Tensor([[[1, 2], [3, 4]]])
    f = Tensor([5, 6])
    g = Tensor([[7, 8]])
    h = dstack(e, f, g)
    h.backward([[[-1, -2, -5, -7], [-3, -4, -6, -8]]])

    assert e.data.shape == e.grad.shape
    assert np.allclose(e.data, np.abs(e.grad))

    assert f.data.shape == f.grad.shape
    assert np.allclose(f.data, np.abs(f.grad))

    assert h.data.shape == h.grad.shape
    assert np.allclose(h.data, np.abs(h.grad))


    i = Tensor([[1, 2], [3, 4]])
    j = Tensor([[[5, 6], [7, 8]], [[9, 10], [11, 12]]])
    k = dstack(i,j)
    k.backward([[[-1, -5, -6], [-2, -7, -8]], [[-3, -9, -10], [-4, -11, -12]]])

    assert i.data.shape == i.grad.shape
    assert np.allclose(i.data, np.abs(i.grad))

    assert j.data.shape == j.grad.shape
    assert np.allclose(j.data, np.abs(j.grad))
