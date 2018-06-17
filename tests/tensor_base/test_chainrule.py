import hypothesis.strategies as st
from numpy.testing import assert_allclose
from hypothesis import given

from mygrad.tensor_base import Tensor


@given(x=st.floats(min_value=-1E-6, max_value=1E6),
       y=st.floats(min_value=-1E-6, max_value=1E6),
       z=st.floats(min_value=-1E-6, max_value=1E6))
def test_chainrule_scalar(x, y, z):
    x = Tensor(x)
    y = Tensor(y)
    z = Tensor(z)

    f = x*y + z
    g = x + z*f*f

    # check side effects
    unused = 2*g - f
    w = 1*f

    g.backward()
    assert_allclose(f.grad, 2 * z.data * f.data)
    assert_allclose(x.grad, 1 + 2 * z.data * f.data * y.data)
    assert_allclose(y.grad, 2 * z.data * f.data * x.data)
    assert_allclose(z.grad, f.data**2 + z.data * 2 * f.data)
    assert w.grad is None
