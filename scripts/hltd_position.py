"""Position coordinates shared by centered closed-loop logging and audits."""
from __future__ import annotations


def centered_node_position(node_index: int, prompt_len: int) -> tuple[int, float]:
    """Map a step=1 centered-field node to its original prompt token."""
    if prompt_len < 4 or not 0 <= node_index < prompt_len - 2:
        raise ValueError(f"Invalid centered node {node_index} for prompt length {prompt_len}.")
    token_index = node_index + 1
    return token_index, token_index / (prompt_len - 1)


def position_phase(position_frac: float) -> str:
    if not 0.0 <= position_frac <= 1.0:
        raise ValueError(f"Invalid normalized position {position_frac}.")
    return "early" if position_frac < 1 / 3 else "middle" if position_frac < 2 / 3 else "late"
