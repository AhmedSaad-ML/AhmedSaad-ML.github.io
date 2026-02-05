"""Experiment configuration for XG-SAFL federated learning."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Central configuration for all experiment hyper-parameters.

    Attributes:
        num_clients: Number of federated clients.
        num_rounds: Total communication rounds.
        local_epochs: Local training epochs per round.
        batch_size: Mini-batch size for local training.
        learning_rate: Optimiser learning rate.
        threshold: Minimum shares needed for secret reconstruction (t).
        seed: Global random seed for reproducibility.
        shap_update_interval: Rounds between SHAP recalculations.
        shap_background_samples: Background dataset size for SHAP.
        shap_test_samples: Test dataset size for SHAP.
        k_ratio: Fraction of features classified as high-precision (top-k).
        modulus_bits: Bit-length of the TJL modulus.
        chunk_bits: Bit-width of packed integer chunks.
        high_bits: Quantisation bits for high-importance features.
        mid_bits: Quantisation bits for medium-importance features.
        low_bits: Quantisation bits for low-importance features.
        groups: Client group assignments (may overlap).
    """

    num_clients: int = 5
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 0.01
    threshold: int = 3
    seed: int = 42
    shap_update_interval: int = 10
    shap_background_samples: int = 100
    shap_test_samples: int = 200
    k_ratio: float = 0.2
    modulus_bits: int = 1024
    chunk_bits: int = 512
    high_bits: int = 24
    mid_bits: int = 16
    low_bits: int = 8
    groups: list[list[int]] = field(
        default_factory=lambda: [[0, 1, 2], [2, 3, 4]]
    )
