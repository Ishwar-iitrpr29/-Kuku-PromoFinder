"""
Module 3A: Cross-Episode Character Alias & ASR Entity Resolver ("LoreLock").
Resolves VLM video-to-script transcription artifacts across the 50 episodes:
1. Distinguishes intentional family houses (House Ashcroft = Juliana's maiden family)
   from 8 phonetic ASR hallucinations of Alexander's Dukedom (Ashbourne <- Ashwin,
   Ashburgh, Ashburn, Ashburton, Ashbridge, Ashwood, Ashburg, Ashbury).
2. Disambiguates speaker-tag collisions (e.g. BLOND NOBLEMAN used for Juliana's Father
   in Ep 2 & 9 vs. villain cousin Rupert in Ep 27, 28, 37, 40-44) and merges visual
   appearance tags (MYSTERIOUS BEARDED MAN -> ALEXANDER, LADY IN GOLD GOWN -> CLARISSA).
3. Flags stage/camera vs. voice tag bugs (OFF-SCREEN VOICE while on-screen) and
   historical era anachronisms ("numbered identification badge" in Regency era).
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from .parser import EpisodeScript, format_mmss


@dataclass
class ContinuityAnomaly:
    episode_number: int
    filename: str
    line_number: int
    timestamp_fmt: str
    category: str  # ASR_TITLE_DRIFT | SPEAKER_COLLISION | VISUAL_ALIAS_TAG | STAGE_DIRECTION_BUG | ERA_ANACHRONISM | PROP_DRIFT
    severity: str  # HIGH | MEDIUM | LOW
    raw_value: str
    canonical_value: str
    context_excerpt: str
    impact_on_promo_and_production: str


# Phonetic ASR drift variants of "Ashbourne" (Alexander's Dukedom), kept strictly
# separate from "Ashcroft" (which is intentionally Juliana's maiden family name).
ASHBOURNE_ASR_VARIANTS: Dict[str, str] = {
    "Ashbury": "Ashbourne (Phonetic ASR drift in Batch 1/2: Ep 20-23)",
    "Ashburg": "Ashbourne (Phonetic ASR drift in Batch 2: Ep 25-27)",
    "Ashwood": "Ashbourne (Phonetic ASR drift in Batch 3: Ep 36)",
    "Ashbridge": "Ashbourne (Phonetic ASR drift in Batch 3: Ep 37)",
    "Ashwin": "Ashbourne (Phonetic ASR surname drift in Ep 39: 'Alexander Ashwin')",
    "Ashburgh": "Ashbourne (Phonetic ASR drift in Batch 4: Ep 40-41)",
    "Ashburn": "Ashbourne (Phonetic ASR drift in Batch 4: Ep 42-43 'Duke Alexander Ashburn')",
    "Ashburton": "Ashbourne (Phonetic ASR drift in Batch 4/5: Ep 45-46 'Duke of Ashburton')",
}

DIRECT_SPEAKER_ALIASES: Dict[str, str] = {
    "JULIANA": "LADY JULIANA ASHCROFT",
    "LADY JULIANA": "LADY JULIANA ASHCROFT",
    "ALEXANDER": "DUKE ALEXANDER ASHBOURNE",
    "MYSTERIOUS BEARDED MAN": "DUKE ALEXANDER ASHBOURNE",
    "THE BEARDED STRANGER": "DUKE ALEXANDER ASHBOURNE",
    "SCARRED STRANGER": "DUKE ALEXANDER ASHBOURNE",
    "CLARISSA": "LADY CLARISSA (SISTER)",
    "LADY CLARISSA": "LADY CLARISSA (SISTER)",
    "LADY IN GOLD GOWN": "LADY CLARISSA (SISTER)",
    "LADY HARRINGTON": "LADY CLARISSA (SISTER)",
    "GERALD": "GERALD HARRINGTON (EX-FIANCE)",
    "THE FORMER GROOM": "GERALD HARRINGTON (EX-FIANCE)",
    "LORD HARRINGTON": "GERALD HARRINGTON (EX-FIANCE)",
    "CATHERINE": "LADY CATHERINE (USURPER BRIDE)",
    "LADY CATHERINE": "LADY CATHERINE (USURPER BRIDE)",
    "RUPERT": "LORD RUPERT ASHBOURNE (USURPER COUSIN)",
    "DOWAGER DUCHESS": "DOWAGER DUCHESS ASHBOURNE (ALEXANDER'S MOTHER)",
    "ALEXANDER'S MOTHER": "DOWAGER DUCHESS ASHBOURNE (ALEXANDER'S MOTHER)",
    "LADY ASHBOURNE": "DOWAGER DUCHESS ASHBOURNE (ALEXANDER'S MOTHER)",
    "LADY ASHCROFT": "LADY ASHCROFT (JULIANA'S STEPMOTHER)",
    "MATRIARCH": "LADY ASHCROFT (JULIANA'S STEPMOTHER)",
    "NOBLE MATRIARCH": "LADY ASHCROFT (JULIANA'S STEPMOTHER)",
    "LORD ASHCROFT": "LORD ASHCROFT (JULIANA'S FATHER)",
    "ELDER NOBLEMAN": "LORD STERLING (CO-CONSPIRATOR)",
    "LORD STERLING": "LORD STERLING (CO-CONSPIRATOR)",
}

ERA_ANACHRONISM_PATTERNS: List[Tuple[re.Pattern, str, str]] = [
    (
        re.compile(r"numbered identification badge", re.I),
        "numbered identification badge",
        "coarse burlap prison smock (19th-century Regency prisoners did not wear modern printed ID badges)",
    ),
    (
        re.compile(r"trench coat", re.I),
        "trench coat",
        "weathered wool military greatcoat (WWI trench coats are a 20th-century anachronism in a Regency dukedom)",
    ),
]


def resolve_speaker_for_episode(raw_speaker: str, episode_number: int, block_text: str = "") -> Tuple[str, str]:
    """
    Returns (canonical_speaker, resolution_reason).
    Handles context-dependent speaker tag collisions across episodes.
    """
    s_up = raw_speaker.upper().strip()
    if not s_up:
        return "", "non_dialogue"

    # Critical collision #1: BLOND NOBLEMAN (53 lines across the dataset)
    if s_up == "BLOND NOBLEMAN":
        if episode_number in (1, 2, 3, 4, 5, 9):
            return (
                "LORD ASHCROFT (JULIANA'S FATHER)",
                "COLLISION_RESOLVED: 'BLOND NOBLEMAN' in Ep 2/9 is Juliana's elderly father, whereas in Ep 27-44 the same VLM tag is used for villain cousin Rupert.",
            )
        else:
            return (
                "LORD RUPERT ASHBOURNE (USURPER COUSIN)",
                "COLLISION_RESOLVED: 'BLOND NOBLEMAN' in Ep 27-44 is usurper cousin Rupert, colliding with Juliana's father's tag from Ep 2/9.",
            )

    # Critical collision #2: Generic 'LORD' in Ep 33-34
    if s_up == "LORD":
        if episode_number in (33, 34):
            return (
                "LORD STERLING (CO-CONSPIRATOR)",
                "ALIAS_RESOLVED: Generic 'LORD' coercing Prudence in Ep 33-34 resolved to co-conspirator Lord Sterling.",
            )

    # Critical bug #3: OFF-SCREEN VOICE in Ep 2 when Lady Ashcroft is on-screen
    if s_up == "OFF-SCREEN VOICE" and episode_number == 2:
        return (
            "LADY ASHCROFT (JULIANA'S STEPMOTHER)",
            "STAGE_BUG_RESOLVED: Tagged as 'OFF-SCREEN VOICE' right after camera cuts to a face-to-face two-shot of the Matriarch.",
        )

    if s_up in DIRECT_SPEAKER_ALIASES:
        canon = DIRECT_SPEAKER_ALIASES[s_up]
        if s_up not in ("JULIANA", "LADY JULIANA", "ALEXANDER", "CLARISSA", "LADY CLARISSA", "GERALD", "CATHERINE", "LADY CATHERINE", "RUPERT", "DOWAGER DUCHESS", "LORD ASHCROFT", "LADY ASHCROFT", "LORD STERLING"):
            return canon, f"VISUAL_DESCRIPTOR_RESOLVED: VLM appearance tag '{s_up}' mapped to canonical character '{canon}'."
        return canon, "canonical"

    return s_up, "unmapped_supporting"


def run_entity_and_continuity_audit(episodes: List[EpisodeScript]) -> Dict:
    """
    Runs across all 50 episodes, enriches every ScriptBlock with its canonical_speaker,
    and compiles a full audit of VLM transcription drift, speaker collisions, and prop/era bugs.
    """
    anomalies: List[ContinuityAnomaly] = []
    ashbourne_variant_counts: Dict[str, List[int]] = {k: [] for k in ASHBOURNE_ASR_VARIANTS}
    ashcroft_mentions: List[int] = []
    speaker_alias_stats: Dict[str, Dict] = {}

    ash_regex = re.compile(r"\b(Ash[a-zA-Z]+)\b")

    for ep in episodes:
        # Check header lines (logline + continuity notes) and blocks for Ash* variants
        header_blob = f"{ep.logline} {ep.continuity_notes}"
        for match in ash_regex.finditer(header_blob):
            word = match.group(1)
            if word.startswith("Ashcroft"):
                if ep.episode_number not in ashcroft_mentions:
                    ashcroft_mentions.append(ep.episode_number)
            elif word in ASHBOURNE_ASR_VARIANTS:
                if ep.episode_number not in ashbourne_variant_counts[word]:
                    ashbourne_variant_counts[word].append(ep.episode_number)

        for blk in ep.blocks:
            # 1. Resolve Speaker Aliases & Collisions
            if blk.block_type == "DIALOGUE" and blk.raw_speaker:
                canon, reason = resolve_speaker_for_episode(blk.raw_speaker, ep.episode_number, blk.text)
                blk.canonical_speaker = canon

                key = f"{blk.raw_speaker} -> {canon}"
                if key not in speaker_alias_stats:
                    speaker_alias_stats[key] = {
                        "raw_tag": blk.raw_speaker,
                        "canonical_character": canon,
                        "line_count": 0,
                        "episodes": set(),
                        "resolution_type": reason.split(":")[0] if ":" in reason else reason,
                    }
                speaker_alias_stats[key]["line_count"] += 1
                speaker_alias_stats[key]["episodes"].add(ep.episode_number)

                if reason.startswith("COLLISION_RESOLVED"):
                    anomalies.append(
                        ContinuityAnomaly(
                            episode_number=ep.episode_number,
                            filename=ep.filename,
                            line_number=blk.line_number,
                            timestamp_fmt=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                            category="SPEAKER_COLLISION",
                            severity="HIGH",
                            raw_value=blk.raw_speaker,
                            canonical_value=canon,
                            context_excerpt=blk.text[:120],
                            impact_on_promo_and_production=reason,
                        )
                    )
                elif reason.startswith("STAGE_BUG_RESOLVED"):
                    anomalies.append(
                        ContinuityAnomaly(
                            episode_number=ep.episode_number,
                            filename=ep.filename,
                            line_number=blk.line_number,
                            timestamp_fmt=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                            category="STAGE_DIRECTION_BUG",
                            severity="HIGH",
                            raw_value=blk.raw_speaker,
                            canonical_value=canon,
                            context_excerpt=blk.text[:120],
                            impact_on_promo_and_production=reason,
                        )
                    )
                elif reason.startswith("VISUAL_DESCRIPTOR_RESOLVED"):
                    anomalies.append(
                        ContinuityAnomaly(
                            episode_number=ep.episode_number,
                            filename=ep.filename,
                            line_number=blk.line_number,
                            timestamp_fmt=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                            category="VISUAL_ALIAS_TAG",
                            severity="MEDIUM",
                            raw_value=blk.raw_speaker,
                            canonical_value=canon,
                            context_excerpt=blk.text[:120],
                            impact_on_promo_and_production=reason,
                        )
                    )

            # 2. Check for Phonetic ASR Title/Dukedom Drift inside block text
            for match in ash_regex.finditer(blk.text):
                word = match.group(1)
                if word.startswith("Ashcroft"):
                    if ep.episode_number not in ashcroft_mentions:
                        ashcroft_mentions.append(ep.episode_number)
                elif word in ASHBOURNE_ASR_VARIANTS:
                    if ep.episode_number not in ashbourne_variant_counts[word]:
                        ashbourne_variant_counts[word].append(ep.episode_number)
                    anomalies.append(
                        ContinuityAnomaly(
                            episode_number=ep.episode_number,
                            filename=ep.filename,
                            line_number=blk.line_number,
                            timestamp_fmt=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                            category="ASR_TITLE_DRIFT",
                            severity="HIGH" if blk.block_type == "DIALOGUE" else "MEDIUM",
                            raw_value=word,
                            canonical_value="Ashbourne",
                            context_excerpt=blk.text[:140],
                            impact_on_promo_and_production=(
                                f"Spoken dialogue/caption calls the Dukedom '{word}' instead of 'Ashbourne'. "
                                "Breaks subtitle consistency when stitching multi-episode promos (e.g. Ep 39 'Ashwin' + Ep 41 'Ashburgh' + Ep 42 'Ashburn' + Ep 45 'Ashburton')."
                            ),
                        )
                    )

            # 3. Check for Historical Era Anachronisms
            for pat, raw_term, fix_desc in ERA_ANACHRONISM_PATTERNS:
                if pat.search(blk.text):
                    anomalies.append(
                        ContinuityAnomaly(
                            episode_number=ep.episode_number,
                            filename=ep.filename,
                            line_number=blk.line_number,
                            timestamp_fmt=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                            category="ERA_ANACHRONISM",
                            severity="MEDIUM",
                            raw_value=raw_term,
                            canonical_value=fix_desc,
                            context_excerpt=blk.text[:140],
                            impact_on_promo_and_production="Visual prompt anachronism in 1800s Regency setting; breaks immersion if copied into AI video regenerations.",
                        )
                    )

    # Format speaker alias table
    alias_summary = []
    for item in speaker_alias_stats.values():
        alias_summary.append({
            "raw_tag": item["raw_tag"],
            "canonical_character": item["canonical_character"],
            "line_count": item["line_count"],
            "episodes": sorted(list(item["episodes"])),
            "resolution_type": item["resolution_type"],
        })
    alias_summary.sort(key=lambda x: x["line_count"], reverse=True)

    return {
        "canonical_show_bible": {
            "bride_maiden_house": {
                "canonical_name": "House Ashcroft (Intentional Separate Family)",
                "members": ["Lady Juliana Ashcroft", "Lord Ashcroft (Father)", "Lady Ashcroft (Stepmother)", "Lady Clarissa Ashcroft/Harrington (Sister)"],
                "episodes_present": sorted(ashcroft_mentions),
                "verification_note": "Verified as intentional separate noble house (Juliana's birth family), NOT a typo for the Dukedom.",
            },
            "groom_dukedom_house": {
                "canonical_name": "Dukedom of Ashbourne",
                "members": ["Duke Alexander Ashbourne", "Dowager Duchess Ashbourne (Mother)", "Lord Rupert Ashbourne (Usurper Cousin)"],
                "asr_phonetic_mutations_detected": {
                    k: {"episodes": v, "note": ASHBOURNE_ASR_VARIANTS[k]}
                    for k, v in ashbourne_variant_counts.items()
                    if v
                },
            },
        },
        "speaker_alias_resolutions": alias_summary,
        "total_anomalies_found": len(anomalies),
        "anomalies_by_category": {
            cat: sum(1 for a in anomalies if a.category == cat)
            for cat in ["ASR_TITLE_DRIFT", "SPEAKER_COLLISION", "VISUAL_ALIAS_TAG", "STAGE_DIRECTION_BUG", "ERA_ANACHRONISM"]
        },
        "anomalies": [asdict(a) for a in anomalies],
    }
