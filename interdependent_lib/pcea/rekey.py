# GPT/Claude generated; context, prompt Erin Spencer
"""
Rekey epoch: produce a new epoch from old Guardian state.

rekey_epoch() derives fresh keys from new IKM, re-seals the live state,
re-wraps the new live key, and splits the new meta key into fresh shares.
Returns a RekeyEpoch record of the operation.

The old keys are not zeroized here; the caller is responsible for
calling wipe() on old key material after confirming the new epoch.
"""

from __future__ import annotations

from .commitment import make_commitment
from .guardian import seal_live_state
from .kdf import derive_keys
from .threshold import split_secret
from .types import LiveState, MetaShares, RekeyEpoch, SealedState, WrappedLiveKey
from .wrap import wrap_live_key


def rekey_epoch(
    old_state: LiveState,
    old_epoch: int,
    new_ikm: bytes,
    new_epoch: int,
    key_id: str,
    guardian_node_id: str,
    sealed_by: str,
    seal_counter: int,
    new_threshold: int,
    sentinels: list[str],
    spectral_snapshot: list | None = None,
) -> tuple[SealedState, WrappedLiveKey, MetaShares, RekeyEpoch]:
    """
    Perform a rekey operation from old_epoch to new_epoch.

    Steps:
    1. Derive fresh (live_key, meta_key) from new_ikm at new_epoch.
    2. Seal old_state under the new live_key at new_epoch.
    3. Wrap new live_key under new meta_key.
    4. Split new meta_key into threshold shares for sentinels.
    5. Compute commitment over new shares.
    6. Return (SealedState, WrappedLiveKey, MetaShares, RekeyEpoch).

    Args:
        old_state:         LiveState to carry forward into the new epoch
        old_epoch:         epoch being superseded
        new_ikm:           fresh input key material for the new epoch
        new_epoch:         new epoch number (must be > old_epoch)
        key_id:            key identifier (may stay the same or rotate)
        guardian_node_id:  identifying string of this Guardian node
        sealed_by:         identity of the sealing party
        seal_counter:      monotonic counter for nonce uniqueness (reset per epoch is safe
                           because ikm changes; counter must be caller-managed)
        new_threshold:     k in k-of-n for new meta shares
        sentinels:         list of n sentinel IDs receiving shares
        spectral_snapshot: optional diagnostic floats for the RekeyEpoch record

    Returns:
        (new_sealed, new_wrapped, new_meta_shares, rekey_record)
    """
    if new_epoch <= old_epoch:
        raise ValueError(f"new_epoch ({new_epoch}) must be > old_epoch ({old_epoch})")

    n = len(sentinels)
    live_key, meta_key = derive_keys(new_ikm, new_epoch, key_id, guardian_node_id)

    # Update state epoch before sealing
    import dataclasses
    new_state = dataclasses.replace(old_state, epoch=new_epoch)

    new_sealed = seal_live_state(
        new_state, live_key, new_epoch, key_id, seal_counter, guardian_node_id, sealed_by
    )
    new_wrapped = wrap_live_key(live_key, meta_key, new_epoch, key_id)

    raw_shares = split_secret(meta_key, new_threshold, n)
    shares_list = [
        {"sentinel_id": sentinels[i], "share": share_bytes, "index": idx}
        for i, (idx, share_bytes) in enumerate(raw_shares)
    ]
    commitment = make_commitment(shares_list)
    new_meta_shares = MetaShares(
        epoch=new_epoch,
        total_shares=n,
        threshold=new_threshold,
        shares=shares_list,
        commitment=commitment,
    )

    rekey_record = RekeyEpoch(
        from_epoch=old_epoch,
        to_epoch=new_epoch,
        new_session_secret_commitment=commitment,
        meta_shares_updated=True,
        renorm_confirmed=False,        # caller sets True after renormalization
        spectral_snapshot=spectral_snapshot or [],
    )

    return new_sealed, new_wrapped, new_meta_shares, rekey_record


def split_meta_key(
    meta_key: bytes,
    threshold: int,
    sentinels: list[str],
) -> MetaShares:
    """
    Split meta_key into threshold-of-n shares for the given sentinels.

    Convenience wrapper used outside of a full rekey (e.g., initial ceremony).
    epoch is carried from context; callers must set MetaShares.epoch afterward
    if needed — this function produces epoch=0 as a placeholder.
    """
    n = len(sentinels)
    raw_shares = split_secret(meta_key, threshold, n)
    shares_list = [
        {"sentinel_id": sentinels[i], "share": share_bytes, "index": idx}
        for i, (idx, share_bytes) in enumerate(raw_shares)
    ]
    commitment = make_commitment(shares_list)
    return MetaShares(
        epoch=0,
        total_shares=n,
        threshold=threshold,
        shares=shares_list,
        commitment=commitment,
    )


def reconstruct_meta_key(shares: list, meta_shares: MetaShares) -> bytes:
    """
    Reconstruct meta_key from a subset of shares.

    The returned bytes are ephemeral — the caller must zeroize after use.
    Verifies the commitment before returning; raises ValueError on mismatch.
    """
    from .commitment import verify_commitment
    from .threshold import reconstruct_secret

    if not verify_commitment(shares, meta_shares.commitment):
        raise ValueError("share commitment verification failed")

    pairs = [(s["index"], s["share"]) for s in shares]
    return reconstruct_secret(pairs)
