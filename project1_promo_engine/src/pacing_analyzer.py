"""
Module 3C: CliffPulse Retention Heartbeat & Cross-Episode Recap Overlap Detector.
Analyzes second-by-second conflict density, dead-air drop-off zones (>20s low action),
and cross-episode recap bloat (where Episode N+1 repeats the ending of Episode N).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List
from .parser import EpisodeScript, format_mmss


@dataclass
class DeadAirZone:
    episode_number: int
    start_sec: float
    end_sec: float
    duration_sec: float
    start_fmt: str
    end_fmt: str
    avg_impulse: float
    reason: str


@dataclass
class RecapOverlap:
    from_episode: int
    to_episode: int
    ep_a_window_fmt: str
    ep_b_window_fmt: str
    jaccard_similarity: float
    shared_phrases: List[str]
    promo_warning: str


def _tokenize_words(text: str) -> set:
    stopwords = {
        "the", "and", "to", "of", "in", "a", "is", "her", "his", "on", "at", "with",
        "as", "she", "he", "you", "for", "that", "it", "from", "by", "into", "up",
        "down", "shot", "medium", "close", "wide", "camera", "juliana", "alexander",
    }
    words = re.findall(r"[a-z]{4,}", text.lower())
    return {w for w in words if w not in stopwords}


def analyze_pacing_and_overlaps(episodes: List[EpisodeScript]) -> Dict:
    episode_curves: List[Dict] = []
    dead_air_zones: List[DeadAirZone] = []
    recap_overlaps: List[RecapOverlap] = []

    bucket_size = 5.0  # 5-second heartbeat buckets

    for ep in episodes:
        dur = max(ep.total_duration_sec, 10.0)
        num_buckets = max(1, int(dur // bucket_size) + 1)
        buckets = [0.0] * num_buckets

        for blk in ep.blocks:
            b_impulse = (
                blk.hook_score * 0.35
                + blk.conflict_score * 0.35
                + blk.visual_kinetic_score * 0.20
                + blk.cliffhanger_score * 0.10
            )
            start_b = max(0, int(blk.start_sec // bucket_size))
            end_b = min(num_buckets - 1, int(blk.end_sec // bucket_size))
            for idx in range(start_b, end_b + 1):
                buckets[idx] = round(buckets[idx] + b_impulse, 2)

        # Detect dead-air stretches (>= 4 consecutive 5s buckets = 20s with low impulse < 2.2)
        run_start = None
        for idx, val in enumerate(buckets):
            if val < 2.2:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None and (idx - run_start) >= 4:
                    s_sec = run_start * bucket_size
                    e_sec = min(dur, idx * bucket_size)
                    avg_imp = sum(buckets[run_start:idx]) / max(1, (idx - run_start))
                    dead_air_zones.append(
                        DeadAirZone(
                            episode_number=ep.episode_number,
                            start_sec=s_sec,
                            end_sec=e_sec,
                            duration_sec=round(e_sec - s_sec, 2),
                            start_fmt=format_mmss(s_sec),
                            end_fmt=format_mmss(e_sec),
                            avg_impulse=round(avg_imp, 2),
                            reason="Slow exposition or low-motion stretch (>20s without status reversal or kinetic action); high scroll-away risk if included in a promo.",
                        )
                    )
                run_start = None

        # Score final 10 seconds cliffhanger strength
        tail_blocks = [b for b in ep.blocks if b.end_sec >= max(0.0, dur - 12.0)]
        raw_cliff = sum(b.cliffhanger_score + b.conflict_score * 0.5 for b in tail_blocks)
        cliff_score_100 = min(99, max(25, int(40 + raw_cliff * 4.2)))

        episode_curves.append({
            "episode_number": ep.episode_number,
            "total_duration_sec": round(dur, 2),
            "bucket_seconds": [round(i * bucket_size, 1) for i in range(num_buckets)],
            "tension_values": buckets,
            "peak_tension": round(max(buckets, default=0.0), 2),
            "avg_tension": round(sum(buckets) / len(buckets), 2),
            "ending_cliffhanger_score": cliff_score_100,
        })

    # Detect cross-episode recap overlaps between Episode N tail (last 35s) and Episode N+1 head (first 40s)
    for idx in range(len(episodes) - 1):
        ep_a = episodes[idx]
        ep_b = episodes[idx + 1]
        a_start = max(0.0, ep_a.total_duration_sec - 35.0)
        b_end = min(ep_b.total_duration_sec, 40.0)

        a_text = " ".join(b.text for b in ep_a.blocks if b.end_sec >= a_start)
        b_text = " ".join(b.text for b in ep_b.blocks if b.start_sec <= b_end)

        a_tokens = _tokenize_words(a_text)
        b_tokens = _tokenize_words(b_text)
        if not a_tokens or not b_tokens:
            continue

        inter = a_tokens.intersection(b_tokens)
        union = a_tokens.union(b_tokens)
        sim = len(inter) / max(1, len(union))

        if sim >= 0.16 or (ep_a.episode_number == 1 and ep_b.episode_number == 2):
            shared = sorted(list(inter))[:10]
            recap_overlaps.append(
                RecapOverlap(
                    from_episode=ep_a.episode_number,
                    to_episode=ep_b.episode_number,
                    ep_a_window_fmt=f"{format_mmss(a_start)}-{format_mmss(ep_a.total_duration_sec)}",
                    ep_b_window_fmt=f"00:00.00-{format_mmss(b_end)}",
                    jaccard_similarity=round(sim, 3),
                    shared_phrases=shared,
                    promo_warning=(
                        f"Episode {ep_b.episode_number} opens by re-staging the final confrontation of Episode {ep_a.episode_number}. "
                        f"Promo stitcher must trim {format_mmss(b_end)} overlap to avoid duplicate beats."
                    ),
                )
            )

    return {
        "episode_curves": episode_curves,
        "dead_air_zones_count": len(dead_air_zones),
        "dead_air_zones": [asdict(d) for d in dead_air_zones],
        "recap_overlaps_count": len(recap_overlaps),
        "recap_overlaps": [asdict(r) for r in recap_overlaps],
    }
