from functools import partial
from numbers import Number

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import reshape
from mygrad.tensor_base import Tensor

from ..custom_strategies import tensors, valid_shapes
from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory


def positional_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, newshape, **kwargs)


def keyword_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, newshape=newshape, **kwargs)


def method_tuple_reshape(arr, newshape, reshaper, **kwargs):
    return arr.reshape(newshape, **kwargs)


def method_unpacked_reshape(arr, newshape, reshaper, **kwargs):
    if newshape == tuple():
        newshape = ((),)
    return (
        arr.reshape(*newshape, **kwargs)
        if isinstance(newshape, tuple)
        else arr.reshape(newshape, **kwargs)
    )


def in_place_reshape(arr, newshape, reshaper, **kwargs):
    to_array = np.asarray if reshaper is np.reshape else Tensor
    arr = +arr  # "touch" array so we can check gradient

    if isinstance(arr, Number):
        arr = to_array(arr)

    arr.shape = newshape
    if isinstance(arr, Tensor) and kwargs.get("constant", False):
        arr._constant = True
    return arr


def test_raising_during_inplace_reshape_doesnt_corrupt_graph():
    x = mg.arange(5.0)
    y = +x
    w = 2 * y
    with pytest.raises(ValueError):
        y.shape = (2, 3)
    w.backward()
    assert_array_equal(w.grad, np.ones_like(w))
    assert_array_equal(y.grad, 2 * np.ones_like(y))
    assert_array_equal(x.grad, 2 * np.ones_like(y))


@given(
    tensor=tensors(), data=st.data(),
)
def test_in_place_reshape(tensor: Tensor, data: st.DataObject):
    assume(tensor.size)

    array = tensor.data.copy()
    newshape = data.draw(valid_shapes(tensor.size, min_len=0), label="newshape")

    tensor.shape = newshape

    array.shape = newshape
    assert_array_equal(array, tensor)

    assert array.base is None
    assert tensor.base is None


@given(
    tensor=tensors(), data=st.data(),
)
def test_in_place_reshape_post_view(tensor: Tensor, data: st.DataObject):
    assume(tensor.size)

    array = tensor.data.copy()
    newshape = data.draw(valid_shapes(tensor.size, min_len=0), label="newshape")

    t1 = tensor[...]
    t1.shape = newshape

    a1 = array[...]
    a1.shape = newshape
    assert_array_equal(array, tensor)
    assert_array_equal(a1, t1)

    assert array.base is None
    assert tensor.base is None
    assert a1.base is array
    assert t1.base is tensor


@pytest.mark.parametrize(
    "reshape_type", [positional_reshape, keyword_reshape],
)
def test_reshape_fwd(reshape_type):
    @fwdprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size)),
    )
    def run_fwd():
        pass

    run_fwd()


@pytest.mark.parametrize(
    "reshape_type", [method_tuple_reshape, method_unpacked_reshape],
)
def test_method_reshape_fwd(reshape_type):
    @fwdprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size)),
        permit_0d_array_as_float=False,
    )
    def run_fwd():
        pass

    run_fwd()


@pytest.mark.parametrize(
    "reshape_type",
    [
        positional_reshape,
        keyword_reshape,
        method_tuple_reshape,
        method_unpacked_reshape,
        in_place_reshape,
    ],
)
def test_reshape_bkwd(reshape_type):
    @backprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size, min_len=0)),
        vary_each_element=True,
    )
    def run_bkwd():
        pass

    run_bkwd()


@pytest.mark.parametrize(
    "bad_input", [tuple(), ((2,), 2), ((2,), 2), (2, (2,)), ((2, (2,)),)]
)
def test_input_validation(bad_input):
    x = Tensor([1, 2])

    with pytest.raises(TypeError):
        x.reshape(*bad_input)


def test_input_validation_matches_numpy():
    try:
        np.reshape(np.array(1), *(1, 1))
    except Exception:
        with pytest.raises(Exception):
            reshape(Tensor(1), *(1, 1))
