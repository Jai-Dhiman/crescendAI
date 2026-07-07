# model/src/follower_bench/segments.py
"""Hard-splice engine: rearranges spans of a clean performance's note
stream into a new timeline, and applies note-level pitch mutations. Every
timeline-rearranging pathology (repeat/jump/restart/hesitation/
tempo_swing) is expressed as a list of Segments; wrong_note is a
NoteMutation applied on top of an identity segment.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerfNote:
    """A single performance note in absolute seconds."""
    onset: float
    offset: float
    pitch: int
    velocity: int


@dataclass(frozen=True)
class Segment:
    """One contiguous span of source (clean) performance time
    [src_start, src_end), replayed starting at dst_start in the new clip,
    optionally time-scaled (dst_duration = (src_end - src_start) *
    time_scale)."""
    src_start: float
    src_end: float
    dst_start: float
    time_scale: float = 1.0

    @property
    def dst_end(self) -> float:
        return self.dst_start + (self.src_end - self.src_start) * self.time_scale


def apply_segments(notes: list[PerfNote], segments: list[Segment]) -> list[PerfNote]:
    """Rebuild a note stream by replaying each segment's source notes
    (onset in the half-open range [src_start, src_end)) at its
    destination time. Segments are applied independently and
    concatenated, then the result is sorted by onset ascending --
    segments may duplicate or omit source notes (repeat/jump), and are
    expected to be given in destination order for a coherent timeline.
    """
    out: list[PerfNote] = []
    for seg in segments:
        for n in notes:
            if seg.src_start <= n.onset < seg.src_end:
                new_onset = seg.dst_start + (n.onset - seg.src_start) * seg.time_scale
                new_duration = (n.offset - n.onset) * seg.time_scale
                out.append(
                    PerfNote(
                        onset=new_onset,
                        offset=new_onset + new_duration,
                        pitch=n.pitch,
                        velocity=n.velocity,
                    )
                )
    out.sort(key=lambda n: n.onset)
    return out
