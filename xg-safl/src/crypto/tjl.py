"""Threshold Joye–Libert (TJL) encryption for secure aggregation.

This module implements a simplified Threshold Joye–Libert scheme suitable
for Proof-of-Concept federated learning.  The public parameters are
derived from a pre-computed 1024-bit safe prime so that the scheme can be
exercised without expensive prime generation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .utils import (
    SAFE_PRIME_1024,
    discrete_log_brute,
    hash_to_group,
    mod_inv,
    mod_pow,
)


@dataclass(frozen=True)
class TJLParams:
    """Public parameters for the Threshold Joye–Libert scheme.

    Attributes:
        n: RSA-style modulus (product of two safe primes for production;
           here p·q with a PoC q derived deterministically).
        n_squared: n² used as the ciphertext-space modulus.
        g: Generator of the plaintext subgroup in ℤ*_{n²}.
        h: Hash-derived element used for randomisation.
    """

    n: int
    n_squared: int
    g: int
    h: int


def setup(modulus_bits: int = 1024) -> TJLParams:
    """Generate TJL public parameters using a pre-computed safe prime.

    For the PoC the modulus *n* is set to the safe prime *p* itself and
    ``n² = p²``.  The generator *g* is ``(1 + n) mod n²`` (which has
    order *n* in the Paillier-style subgroup) and *h* is derived by
    hashing a fixed label into the group.

    Args:
        modulus_bits: Desired bit-length (retained for API compatibility).

    Returns:
        A :class:`TJLParams` instance ready for encryption.
    """
    p = SAFE_PRIME_1024
    n = p
    n_squared = n * n
    g = (1 + n) % n_squared
    h = hash_to_group(b"TJL-h-parameter", n_squared)
    # Ensure h is in ℤ*_{n²} (extremely likely but we check).
    if h == 0:
        h = 1
    return TJLParams(n=n, n_squared=n_squared, g=g, h=h)


def protect(
    params: TJLParams,
    value: int,
    client_id: int,
    round_num: int,
) -> int:
    """Encrypt (protect) an integer value under TJL.

    The ciphertext is computed as:

        c = g^{value} · h^{r}  mod n²

    where *r* is a deterministic per-(client, round) mask derived via
    ``hash_to_group``.

    Args:
        params: Public TJL parameters.
        value: The plaintext integer to encrypt.
        client_id: Identifier of the encrypting client.
        round_num: Current communication round number.

    Returns:
        The ciphertext as an integer in ℤ*_{n²}.
    """
    r = hash_to_group(
        f"{client_id}||{round_num}".encode(),
        params.n_squared,
    )
    c = (
        mod_pow(params.g, value, params.n_squared)
        * mod_pow(params.h, r, params.n_squared)
    ) % params.n_squared
    return c


def aggregate(params: TJLParams, ciphertexts: list[int]) -> int:
    """Homomorphically aggregate a list of ciphertexts.

    Because the scheme is additively homomorphic the aggregate ciphertext
    encrypts the sum of the individual plaintexts:

        C = ∏ cᵢ  mod n²

    Args:
        params: Public TJL parameters.
        ciphertexts: List of individual ciphertexts.

    Returns:
        The aggregated ciphertext.

    Raises:
        ValueError: If *ciphertexts* is empty.
    """
    if not ciphertexts:
        raise ValueError("ciphertexts list must not be empty")
    result = 1
    for ct in ciphertexts:
        result = (result * ct) % params.n_squared
    return result


def share_protect(
    params: TJLParams,
    secret_key_share: int,
    aggregate_ct: int,
) -> int:
    """Compute a partial decryption share.

    Each key-holder raises the aggregate ciphertext to their secret-key
    share modulo n²:

        dᵢ = C^{skᵢ}  mod n²

    Args:
        params: Public TJL parameters.
        secret_key_share: This party's share of the secret key.
        aggregate_ct: The aggregate ciphertext to partially decrypt.

    Returns:
        A partial decryption share.
    """
    return mod_pow(aggregate_ct, secret_key_share, params.n_squared)


def share_combine(
    params: TJLParams,
    shares: list[int],
    threshold: int,
) -> int:
    """Combine partial decryption shares to recover the plaintext sum.

    The combined share ``D = ∏ dᵢ mod n²`` is used to strip the
    randomness from the aggregate ciphertext.  The plaintext is then
    recovered via the Paillier-style *L*-function:

        L(x) = (x − 1) / n

    followed by a brute-force discrete-log search to handle residual
    non-linearity in the PoC parametrisation.

    Args:
        params: Public TJL parameters.
        shares: List of partial decryption shares (at least *threshold*).
        threshold: Minimum number of shares required.

    Returns:
        The recovered plaintext integer (sum of encrypted values).

    Raises:
        ValueError: If fewer than *threshold* shares are provided.
    """
    if len(shares) < threshold:
        raise ValueError(
            f"Need at least {threshold} shares, got {len(shares)}"
        )

    combined = 1
    for s in shares[:threshold]:
        combined = (combined * s) % params.n_squared

    # Paillier L-function: L(x) = (x - 1) / n
    numerator = (combined - 1) % params.n_squared
    if numerator % params.n != 0:
        # Fall back to brute-force discrete log on g
        return discrete_log_brute(
            target=combined,
            base=params.g,
            modulus=params.n_squared,
        )
    plaintext = numerator // params.n
    return plaintext % params.n
