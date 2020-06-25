from mygrad.tensor_creation.funcs import identity
from .constant import constant
from .glorot_normal import glorot_normal
from .normal import normal

__all__ = [
    "constant",
    "glorot_normal",
    "identity",
    "normal",
]
