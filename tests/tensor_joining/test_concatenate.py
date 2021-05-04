from mygrad.tensor_base import Tensor
from mygrad import concatenate

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data())
def test_concatenate_fwd(data):
    dim = data.draw(st.integers(0, 4))
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, min_dims=dim+1, max_dims=dim+2),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))
    c = Tensor(a)

    result = np.concatenate((a, b), axis=dim)
    o = concatenate(c, b, axis=dim)

    assert isinstance(o, Tensor)
    assert np.allclose(o.data, result)


@given(st.data())
def test_concatenate_backward(data):
    dim = np.random.randint(0, 4)
    a = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, min_dims=dim+1, max_dims=dim+2),
                             dtype=float,
                             elements=st.floats(0, 100)))
    b = data.draw(hnp.arrays(shape=a.shape,
                             dtype=float,
                             elements=st.floats(0, 100)))
    a = Tensor(a)
    b = Tensor(b)
    c = concatenate(a, b, axis=dim)

    g = -c.data
    c.backward(g)

    assert a.data.shape == a.grad.shape
    assert np.allclose(a.data, np.abs(a.grad))

    assert b.data.shape == b.grad.shape
    assert np.allclose(b.data, np.abs(b.grad))
