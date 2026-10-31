# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Experimental remote-M reuse for the existing contiguous-rank CP layout.

Forward still gathers (h, M). Backward gathers only dh, reusing the forward M
of intervening ranks. This protocol is independent of the local AutoCP split.
"""

import torch
import torch.distributed as dist

from ..cp.comm import all_gather_into_tensor, pack_hm, unpack_hm


def correct_fwd_with_m_cache(h_seq, m_seq, initial_state, cp_context, state_v_first):
    from .cp_fwd import correct_initial_states

    rank = dist.get_rank(cp_context.group)
    pre, post = cp_context.pre_num_ranks, cp_context.post_num_ranks
    hm = pack_hm(h_seq[-1], m_seq[-1])
    gathered, _ = all_gather_into_tensor(hm, group=cp_context.group)

    # Copy M out of the packed receive buffer so the h payload is not retained.
    # Include the two unused endpoint slots to match the correction kernel's
    # rank indexing. No cache is needed when this rank has no following ranks.
    split = h_seq[-1].numel() * h_seq.element_size()
    remote_m = (
        gathered[rank:rank + post + 1, split:].contiguous().view(m_seq.dtype)
        .reshape(post + 1, *m_seq.shape[1:])
        if post else m_seq.new_empty((0, *m_seq.shape[1:]))
    )
    if cp_context.is_first_rank:
        return initial_state, remote_m

    h_buf, m_buf = unpack_hm(gathered[rank - pre:rank + 1], h_seq[-1], m_seq[-1])
    seq_map, fallback = cp_context.get_fwd_scan_tensors(h_seq.shape[1])
    states = correct_initial_states(
        raw_h0=None, ht_buffer=h_buf, mt_buffer=m_buf, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    card_h0 = torch.zeros_like(h_seq, dtype=torch.float32) if initial_state is None else initial_state.clone()
    card_h0[0] = states[-1]
    return card_h0, remote_m


def correct_bwd_with_m_cache(dh_seq, remote_m, dht, cp_context, state_v_first):
    from .cp_fwd import correct_terminal_states

    # Every rank participates, including ranks with no cross-rank dependency.
    gathered, _ = all_gather_into_tensor(dh_seq[0].contiguous(), group=cp_context.group)
    if cp_context.is_last_rank:
        return dht

    rank = dist.get_rank(cp_context.group)
    post = cp_context.post_num_ranks
    dh_buf = gathered[rank:rank + post + 1]
    seq_map, fallback = cp_context.get_bwd_scan_tensors(dh_seq.shape[1])
    fallback = fallback.clone()  # Do not modify the context's cached baseline mask.
    # The end rank's first-sequence dh already includes its terminal seed. Start
    # from that dh directly: its last-sequence M can belong to another sequence.
    # All intervening ranks contain only the spanning sequence, so their first
    # and last M are identical. Our own M is never consumed by the exclusive scan.
    fallback[-1] = False
    states = correct_terminal_states(
        raw_dht=None, dht_buffer=dh_buf, mt_buffer=remote_m, fallback_mask=fallback,
        seq_map_r2c=seq_map, state_v_first=state_v_first,
    )
    card_dht = torch.zeros_like(dh_seq, dtype=torch.float32) if dht is None else dht.clone()
    card_dht[-1] = states[0]
    return card_dht
