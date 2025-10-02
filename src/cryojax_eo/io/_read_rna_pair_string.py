import jax.numpy as jnp
from jaxtyping import Array, Int


def read_rna_pair_string(dot_bracket: str) -> Int[Array, "n_pairs 2"]:
    """Convert dot-bracket notation to base pair indices.
    Args:
        dot_bracket (str): Dot-bracket notation string.
    Returns:
        jnp.ndarray: Array of shape (n_pairs, 2) containing indices of base pairs.
    """
    # Initialize an empty stack and a list to store pairs
    stack = []
    pairs = []

    # Traverse the dot-bracket notation
    for i, char in enumerate(dot_bracket):
        if char == "(":
            # Push the index of '(' onto the stack
            stack.append(i)
        elif char == ")":
            # Pop from the stack to get the matching '(' index
            start_index = stack.pop()
            # Store the pair (start_index, end_index)
            pairs.append((start_index, i))

    return jnp.array(pairs)
