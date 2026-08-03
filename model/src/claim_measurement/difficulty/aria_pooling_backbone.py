"""Aria-medium pooling control (#138 Phase 0 follow-up).

The bake-off showed MoonBeam-839M/mean_pool beating Aria by +0.054 tau-c, but
~85% of that gap was MoonBeam's own mean-over-tokens vs last-token pooling, not
the backbone. This backbone isolates the confound: it runs the SAME
aria-medium-embedding checkpoint over the SAME chunks as the deployed
`get_global_embedding_from_midi` path and returns BOTH poolings from one
forward pass.

    eos_pool  -- reproduces the shipped behaviour (chunk EOS position, chunks
                 averaged). Must land on the bake-off's Aria number; it is the
                 harness's own control.
    mean_pool -- mean over the chunk's tokens, chunks averaged. The only thing
                 that differs from eos_pool.

Chunking (300 notes/chunk, 2048-token cap, EOS forced back after truncation)
is replicated from aria.embedding rather than reimplemented, so the comparison
cannot drift from what the deployed path does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class AriaPoolingBackbone:
    """Backbone protocol implementation returning eos_pool + mean_pool."""

    NOTES_PER_CHUNK = 300

    def embed(self, midi_path: Path) -> dict:
        import torch
        from aria.embedding import (
            MAX_EMBEDDING_SEQ_LEN,
            _get_chunks,
            _validate_midi_for_emb,
        )
        from ariautils.midi import MidiDict
        from ariautils.tokenizer import AbsTokenizer

        from model_improvement.aria_embeddings import _load_embedding_model

        model = _load_embedding_model()
        tokenizer = AbsTokenizer()

        midi_dict = MidiDict.from_midi(mid_path=str(midi_path))
        _validate_midi_for_emb(midi_dict)
        chunks = _get_chunks(midi_dict=midi_dict, notes_per_chunk=self.NOTES_PER_CHUNK)

        seqs = []
        for chunk in chunks:
            seq = tokenizer.tokenize(chunk, add_dim_tok=False)[:MAX_EMBEDDING_SEQ_LEN]
            if seq[-1] != tokenizer.eos_tok:
                seq[-1] = tokenizer.eos_tok
            seqs.append(seq)

        eos_embs, mean_embs = [], []
        with torch.no_grad():
            for seq in seqs:
                eos_pos = seq.index(tokenizer.eos_tok)
                enc = torch.tensor(tokenizer.encode(seq))
                hidden = model.forward(enc.view(1, -1))[0]  # (seq_len, emb_size)
                eos_embs.append(hidden[eos_pos])
                # Mean over the chunk's real tokens, up to and including EOS --
                # anything past eos_pos is padding-side noise the EOS variant
                # never sees, and including it would make this a comparison of
                # two different token sets rather than of two poolings.
                mean_embs.append(hidden[: eos_pos + 1].mean(dim=0))

        return {
            "eos_pool": torch.stack(eos_embs).mean(dim=0).float().numpy().astype(np.float32),
            "mean_pool": torch.stack(mean_embs).mean(dim=0).float().numpy().astype(np.float32),
        }
