from typing import Any, Callable, Union
from typing_extensions import TypeAlias

from jaxtyping import Array, Float


PerParticleArgs: TypeAlias = Union[Float, None, Any]
Image = Float[Array, "y x"]

LossFn: TypeAlias = Callable[[Image, Image, PerParticleArgs], Float]
# LossFnNoArgs: TypeAlias = Callable[[Image, Image], Float]

# LossFn: TypeAlias = Union[LossFnArgs, LossFnNoArgs]
