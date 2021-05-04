from mygrad.tensor_base import Tensor
from mygrad import hstack

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_hstack_fwd(data):
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))
    c = Tensor(a)
    b = b

    result = np.hstack((a, b))
    o = hstack(c, b)

    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


def test_hstack_backward():
    a = Tensor([3])
    b = Tensor([1, 2, 3])
    c = Tensor(2)
    d = hstack(a, b, c)
    d.backward([-3, -1, -2, -3, -2])

    assert a.data.shape == a.grad.shape
    assert np.allclose(a.data, np.abs(a.grad))

    assert b.data.shape == b.grad.shape
    assert np.allclose(b.data, np.abs(b.grad))

    assert c.data.shape == c.grad.shape
    assert np.allclose(c.data, np.abs(c.grad))


    e = Tensor([[1, 2], [3, 4]])
    f = Tensor([[5], [6]])
    g = hstack(e, f)
    g.backward([[-1, -2, -5], [-3, -4, -6]])

    assert e.data.shape == e.grad.shape
    assert np.allclose(e.data, np.abs(e.grad))

    assert g.data.shape == g.grad.shape
    assert np.allclose(g.data, np.abs(g.grad))
