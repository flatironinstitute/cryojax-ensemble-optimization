from typing import Callable, TypeVar, Union, Any
from typing_extensions import TypeAlias
from jaxtyping import Float, Array

PerParticleArgs: TypeAlias = Union[Float, None, Any]
Image = Float[Array, "y x"]

LossFn: TypeAlias = Callable[[Image, Image, PerParticleArgs], Float]
#LossFnNoArgs: TypeAlias = Callable[[Image, Image], Float]

#LossFn: TypeAlias = Union[LossFnArgs, LossFnNoArgs]