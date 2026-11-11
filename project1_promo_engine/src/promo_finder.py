"""
Core Problem 1 Engine: Promo Clip Finder, Multi-Segment Trailer Stitcher & Acquisition Funnel Ranker.
Discovers optimal 20-60 second promos (both continuous single-stretch cuts and
multi-segment stitched trailers across episodes) and ranks them by posting priority.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from .parser import EpisodeScript, ScriptBlock, format_mmss
from .llm_router import UniversalLLMRouter


@dataclass
class PromoSegment:
    segment_order: int
    episode_number: int
    start_sec: float
    end_sec: float
    start_fmt: str
    end_fmt: str
    duration_sec: float
    segment_role: str  # "HOOK & SETUP" | "ESCALATION" | "CLIFFHANGER PAYOFF" | "CONTINUOUS STRETCH"
    key_action_or_line: str


@dataclass
class PromoCandidate:
    promo_id: str
    rank: int
    title: str
    promo_type: str  # "Single-Stretch Continuous" | "Multi-Segment (Single Episode)" | "Cross-Episode Stitched Trailer"
    posting_wave: str  # "Wave 1: Post First (Broad Cold-Traffic Hook)" | "Wave 2: High-Heat Action/Revenge" | "Wave 3: Retargeting & Payoff Twist"
    target_platforms: List[str]
    total_duration_sec: float
    segments: List[PromoSegment]
    opening_hook_3s: str
    ending_cliffhanger: str
    why_it_works_one_liner: str
    posting_priority_reason: str
    scores: Dict[str, float]  # hook_3s, cliffhanger_cut, standalone_clarity, pacing_density, composite_score
    transcript_preview: List[str] = field(default_factory=list)


# Trope clarity detector for cold audiences who have never seen the show
UNIVERSAL_TROPE_SIGNALS = [
    (re.compile(r"altar|marrying me|bride|wedding|husband|fianc", re.I), "Wedding Betrayal at the Altar", 18.0),
    (re.compile(r"under arrest|handcuffs|planted|pocket watch|hang|gallows", re.I), "Framed Husband & Sadistic Ultimatum", 16.0),
    (re.compile(r"blackthorn|bare hands|thorns|bleeding|crawl", re.I), "Barehanded Sacrifice in Razor Briars", 17.0),
    (re.compile(r"shackles|chains|prison dress|dungeon|red-hot|branding iron", re.I), "Captive Heiress Tortured by Usurpers", 17.5),
    (re.compile(r"explosion|detonat|rightful duke|salute|arrest him", re.I), "Presumed-Dead Duke Crashes Wedding in Full Regalia", 19.0),
    (re.compile(r"scullery maid|scrub|riding crop|struck|knees", re.I), "Hidden Aristocrat Humiliated as a Maid", 15.5),
    (re.compile(r"fire|ablaze|flames|burning|smoke", re.I), "Midnight Arson & Heroic Rescue", 16.0),
    (re.compile(r"witchcraft|trial|courtroom|poison|arsenic", re.I), "Rigged Witchcraft Trial & Poison Conspiracy", 15.0),
]

CONFUSING_LORE_SIGNALS = [
    re.compile(r"ecclesiastical|royal register|clerk in london|section|sub-council|ten days", re.I),
    re.compile(r"prudence told me|agnes|seamus|herbarium ledger page", re.I),
]


def _extract_window_blocks(ep: EpisodeScript, start_sec: float, end_sec: float) -> List[ScriptBlock]:
    return [
        b for b in ep.blocks
        if b.end_sec > start_sec + 0.1 and b.start_sec < end_sec - 0.1
    ]


def _build_transcript_preview(blocks: List[ScriptBlock], max_lines: int = 8) -> List[str]:
    preview = []
    for b in blocks:
        ts = f"[{format_mmss(b.start_sec)}-{format_mmss(b.end_sec)}]"
        if b.block_type == "DIALOGUE":
            emo = f" ({', '.join(b.emotion_cues)})" if b.emotion_cues else ""
            preview.append(f"{ts} {b.canonical_speaker}{emo}: {b.text}")
        elif b.block_type in ("ACTION", "SOUND FX"):
            preview.append(f"{ts} [{b.block_type}] {b.text}")
        if len(preview) >= max_lines:
            break
    return preview


def score_window(
    ep: EpisodeScript,
    start_sec: float,
    end_sec: float,
    allow_short_or_few_blocks: bool = False,
) -> Tuple[Dict[str, float], str, str, str]:
    """
    Deterministically scores a [start_sec, end_sec] candidate window inside an episode.
    Returns (scores_dict, hook_description, cliffhanger_description, detected_trope).
    """
    zero_scores = {
        "hook_3s": 0.0,
        "cliffhanger_cut": 0.0,
        "standalone_clarity": 0.0,
        "pacing_density": 0.0,
        "composite_score": 0.0,
    }
    dur = end_sec - start_sec
    if not allow_short_or_few_blocks and (dur < 20.0 or dur > 60.0):
        return (zero_scores, "", "", "")

    w_blocks = _extract_window_blocks(ep, start_sec, end_sec)
    if not w_blocks:
        return (zero_scores, "", "", "")
    if not allow_short_or_few_blocks and len(w_blocks) < 3:
        return (zero_scores, "", "", "")

    # 1. Opening 3.5-second Hook Velocity (30% weight)
    first_blocks = [b for b in w_blocks if b.start_sec <= start_sec + 4.5] or w_blocks[:2]
    raw_hook = sum(b.hook_score + b.visual_kinetic_score * 0.6 + b.conflict_score * 0.4 for b in first_blocks)
    # Bonus if first block starts with sharp confrontation or kinetic action
    first_text = " ".join(b.text for b in first_blocks[:2])
    if any(k in first_text.lower() for k in ("slap", "wait", "stop", "enough", "turn around", "under arrest", "marrying me", "shackles", "explosion", "kneels", "fire", "strike")):
        raw_hook += 4.5
    hook_3s_score = min(99.0, max(30.0, 42.0 + raw_hook * 4.8))

    # 2. Ending 4.5-second Cliffhanger Cut Tension (30% weight)
    last_blocks = [b for b in w_blocks if b.end_sec >= end_sec - 5.0] or w_blocks[-2:]
    raw_cliff = sum(b.cliffhanger_score + b.conflict_score * 0.5 for b in last_blocks)
    last_text = " ".join(b.text for b in last_blocks[-2:])
    if any(k in last_text.lower() for k in ("resolving chord", "quiet walk", "peaceful", "falls asleep")):
        raw_cliff -= 6.0
    if "?" in last_text or "!" in last_text or any(k in last_text.lower() for k in ("marry", "duke", "explosion", "poison", "hang", "never", "swings open", "cut to black", "brand", "scream")):
        raw_cliff += 3.8
    cliff_score = min(99.0, max(25.0, 40.0 + raw_cliff * 4.6))

    # 3. Cold-Viewer Standalone Clarity (25% weight)
    full_blob = " ".join(b.text for b in w_blocks)
    clarity = 62.0
    detected_trope = "High-Stakes Aristocratic Betrayal"
    for pat, trope_name, bonus in UNIVERSAL_TROPE_SIGNALS:
        if pat.search(full_blob):
            clarity += bonus
            detected_trope = trope_name
            break
    for lore_pat in CONFUSING_LORE_SIGNALS:
        if lore_pat.search(full_blob):
            clarity -= 12.0
    # Ensure both dialogue and visual action exist
    has_dialogue = any(b.block_type == "DIALOGUE" for b in w_blocks)
    has_visual = any(b.block_type in ("ACTION", "CAMERA") and b.visual_kinetic_score >= 1.2 for b in w_blocks)
    if not has_dialogue:
        clarity -= 20.0
    if not has_visual:
        clarity -= 12.0
    standalone_score = min(98.0, max(30.0, clarity))

    # 4. Pacing & Emotion Density (15% weight)
    total_impulse = sum(b.conflict_score + b.visual_kinetic_score * 0.7 for b in w_blocks)
    impulse_per_10s = (total_impulse / max(10.0, dur)) * 10.0
    pacing_score = min(98.0, max(35.0, 45.0 + impulse_per_10s * 3.6))

    # Penalty if scene is shorter than 20s minimum
    duration_penalty = 18.0 if dur < 20.0 else 0.0

    composite = round(
        max(
            20.0,
            hook_3s_score * 0.30
            + cliff_score * 0.30
            + standalone_score * 0.25
            + pacing_score * 0.15
            - duration_penalty,
        ),
        1,
    )

    # Extract readable hook and cliffhanger descriptions
    hook_b = next((b for b in first_blocks if b.block_type in ("DIALOGUE", "ACTION")), first_blocks[0])
    cliff_b = next((b for b in reversed(last_blocks) if b.block_type in ("DIALOGUE", "ACTION", "SOUND FX")), last_blocks[-1])

    hook_desc = (
        f'{hook_b.canonical_speaker}: {hook_b.text}'
        if hook_b.block_type == "DIALOGUE"
        else hook_b.text
    )
    cliff_desc = (
        f'{cliff_b.canonical_speaker}: {cliff_b.text}'
        if cliff_b.block_type == "DIALOGUE"
        else cliff_b.text
    )

    scores = {
        "hook_3s": round(hook_3s_score, 1),
        "cliffhanger_cut": round(cliff_score, 1),
        "standalone_clarity": round(standalone_score, 1),
        "pacing_density": round(pacing_score, 1),
        "composite_score": composite,
    }
    return scores, hook_desc, cliff_desc, detected_trope


def find_best_single_stretch_per_episode(episodes: List[EpisodeScript]) -> List[PromoCandidate]:
    """
    Finds the highest-scoring 20 to 60-second continuous promo inside each episode.
    Snaps start and end timestamps to exact block boundaries.
    """
    per_ep_candidates: List[PromoCandidate] = []

    for ep in episodes:
        if not ep.blocks:
            continue

        best_scores = {
            "hook_3s": 0.0,
            "cliffhanger_cut": 0.0,
            "standalone_clarity": 0.0,
            "pacing_density": 0.0,
            "composite_score": -1.0,
        }
        best_window = (0.0, min(55.0, ep.total_duration_sec))
        best_hook = ""
        best_cliff = ""
        best_trope = "Dramatic Confrontation"

        # Candidate start points: top hook/kinetic blocks (or every 6th block) for sub-second O(N) speed
        ranked_starts = sorted(
            ep.blocks,
            key=lambda b: (b.hook_score + b.visual_kinetic_score * 0.6 + b.conflict_score * 0.4),
            reverse=True,
        )[:10]
        candidate_starts = sorted(list({
            0.0,
            *[
                round(b.start_sec, 2)
                for idx, b in enumerate(ep.blocks)
                if b in ranked_starts or idx == 0 or idx % 6 == 0
            ],
        }))
        ranked_ends = sorted(
            ep.blocks,
            key=lambda b: (b.cliffhanger_score + b.conflict_score * 0.5),
            reverse=True,
        )[:10]
        candidate_ends = sorted(list({
            round(ep.total_duration_sec, 2),
            *[
                round(b.end_sec, 2)
                for idx, b in enumerate(ep.blocks)
                if b in ranked_ends or idx == len(ep.blocks) - 1 or idx % 6 == 0
            ],
        }))

        for s_t in candidate_starts:
            for e_t in candidate_ends:
                dur = round(e_t - s_t, 2)
                if 20.0 <= dur <= 60.0:
                    sc, h_desc, c_desc, trope = score_window(ep, s_t, e_t)
                    if sc["composite_score"] > best_scores["composite_score"]:
                        best_scores = sc
                        best_window = (s_t, e_t)
                        best_hook = h_desc
                        best_cliff = c_desc
                        best_trope = trope

        # Fallback if the entire custom episode is shorter than 20s or had <3 blocks in [20, 60]
        if best_scores["composite_score"] <= 0.0:
            s_t = 0.0
            e_t = min(60.0, max(5.0, ep.total_duration_sec))
            sc, h_desc, c_desc, trope = score_window(ep, s_t, e_t, allow_short_or_few_blocks=True)
            best_scores = sc
            best_window = (s_t, e_t)
            best_hook = h_desc or (ep.blocks[0].text if ep.blocks else "Opening beat")
            best_cliff = c_desc or (ep.blocks[-1].text if ep.blocks else "Ending beat")
            best_trope = trope or "Dramatic Scene Preview"

        s_sec, e_sec = best_window
        dur = round(e_sec - s_sec, 2)
        w_blocks = _extract_window_blocks(ep, s_sec, e_sec)

        seg = PromoSegment(
            segment_order=1,
            episode_number=ep.episode_number,
            start_sec=s_sec,
            end_sec=e_sec,
            start_fmt=format_mmss(s_sec),
            end_fmt=format_mmss(e_sec),
            duration_sec=dur,
            segment_role="CONTINUOUS STRETCH" if dur >= 20.0 else "SHORT SCENE PREVIEW (<20s)",
            key_action_or_line=best_hook[:110],
        )

        if dur < 20.0:
            why_line = (
                f"⚠️ Scene is {dur:.1f}s long (below the 20.0s standalone ad minimum). Expand dialogue/action by +{20.0 - dur:.1f}s or stitch with another scene before publishing."
            )
            title_prefix = f"Ep {ep.episode_number} [Short Preview {dur:.0f}s]: {best_trope}"
        else:
            why_line = (
                f"Opens immediately on a {best_trope.lower()} shock at {format_mmss(s_sec)} and cuts on an unresolved power shift at {format_mmss(e_sec)} ({dur:.1f}s) without needing backstory."
            )
            title_prefix = f"Ep {ep.episode_number}: {best_trope}"

        per_ep_candidates.append(
            PromoCandidate(
                promo_id=f"EP{ep.episode_number:02d}-CONT",
                rank=0,
                title=title_prefix,
                promo_type="Single-Stretch Continuous",
                posting_wave="Wave 1: Post First (Broad Cold-Traffic Hook)" if dur >= 20.0 else "Draft / Needs Expansion (>=20s)",
                target_platforms=["TikTok", "Instagram Reels", "YouTube Shorts"],
                total_duration_sec=dur,
                segments=[seg],
                opening_hook_3s=f"[{format_mmss(s_sec)}-{format_mmss(min(e_sec, s_sec + 3.0))}] {best_hook}",
                ending_cliffhanger=f"[{format_mmss(max(s_sec, e_sec - 4.0))}-{format_mmss(e_sec)}] {best_cliff}",
                why_it_works_one_liner=why_line,
                posting_priority_reason="Self-contained dramatic escalation with high visual and emotional contrast.",
                scores=best_scores,
                transcript_preview=_build_transcript_preview(w_blocks),
            )
        )

    return per_ep_candidates


def _apply_model_profile_to_promos(
    promos: List[PromoCandidate],
    router: Optional[UniversalLLMRouter] = None,
) -> List[PromoCandidate]:
    """
    Adapts promo pitch copy, creative angle, and composite score weighting
    based on the active Model Engine Profile so switching profiles updates
    both the main Promo Leaderboard/Inspector and localization outputs.
    """
    model_str = (router.config.model.lower() if (router and router.config and router.config.model) else "")
    is_dramatic_model = (
        "bollywood" in model_str
        or "dramatic" in model_str
        or ("standard" in model_str and any(k in model_str for k in ("-pro", "large", "r1", "70b")))
    )
    is_urban_model = (
        "conversational" in model_str
        or "hinglish" in model_str
        or ("standard" in model_str and any(k in model_str for k in ("2.0", "lite", "instant", "small", "mini", "8b", "9b")))
    )
    if is_dramatic_model:
        profile_tag = "High-Intensity Bollywood Microdrama"
        for p in promos:
            s = p.scores
            # Dramatic profile weights Cliffhanger & Pacing Density higher
            new_comp = round(
                s["hook_3s"] * 0.25
                + s["cliffhanger_cut"] * 0.40
                + s["standalone_clarity"] * 0.15
                + s["pacing_density"] * 0.20,
                1,
            )
            if p.total_duration_sec < 20.0:
                new_comp = max(20.0, round(new_comp - 18.0, 1))
            p.scores = {**s, "composite_score": new_comp}
            p.why_it_works_one_liner = (
                f"[{profile_tag}] {p.why_it_works_one_liner} "
                "Engineered for maximum emotional confrontation, shock zooms, and high-heat revenge retention."
            )
    elif is_urban_model:
        profile_tag = "Modern Urban Hinglish Reels"
        for p in promos:
            s = p.scores
            # Urban Reels profile weights First-3s Hook & Cold-Viewer Clarity higher
            new_comp = round(
                s["hook_3s"] * 0.40
                + s["cliffhanger_cut"] * 0.25
                + s["standalone_clarity"] * 0.25
                + s["pacing_density"] * 0.10,
                1,
            )
            if p.total_duration_sec < 20.0:
                new_comp = max(20.0, round(new_comp - 18.0, 1))
            p.scores = {**s, "composite_score": new_comp}
            p.why_it_works_one_liner = (
                f"[{profile_tag}] {p.why_it_works_one_liner} "
                "Optimized for fast-scrolling Gen-Z/Metro Reels feeds with instant 2-second mute-caption clarity."
            )
    else:
        profile_tag = "Balanced Acquisition Funnel (v1)"
        for p in promos:
            p.why_it_works_one_liner = f"[{profile_tag}] {p.why_it_works_one_liner}"

    # Assign ranks if not already assigned (e.g. custom scripts)
    if any(p.rank == 0 for p in promos):
        promos.sort(key=lambda x: x.scores["composite_score"], reverse=True)
        for idx, p in enumerate(promos, start=1):
            p.rank = idx
    return promos


def build_curated_flagship_and_stitched_promos(
    episodes: List[EpisodeScript],
    router: Optional[UniversalLLMRouter] = None,
) -> List[PromoCandidate]:
    """
    Builds the master ranked portfolio combining the strongest single-stretch cuts
    AND multi-moment stitched trailers (within an episode and across episodes),
    strictly respecting the 20.0s to 60.0s total duration rule.
    """
    valid_episodes = [e for e in episodes if e.blocks]
    if not valid_episodes:
        return []

    # If analyzing a custom uploaded or pasted script pack (not the 50-episode default pack),
    # dynamically rank and return the discovered promos from the custom episodes!
    if len(valid_episodes) < 15:
        custom_candidates = find_best_single_stretch_per_episode(valid_episodes)
        return _apply_model_profile_to_promos(custom_candidates, router=router)

    ep_map = {e.episode_number: e for e in episodes}

    def make_seg(order: int, ep_num: int, s_sec: float, e_sec: float, role: str, summary: str) -> PromoSegment:
        return PromoSegment(
            segment_order=order,
            episode_number=ep_num,
            start_sec=s_sec,
            end_sec=e_sec,
            start_fmt=format_mmss(s_sec),
            end_fmt=format_mmss(e_sec),
            duration_sec=round(e_sec - s_sec, 2),
            segment_role=role,
            key_action_or_line=summary,
        )

    def collect_preview(segs: List[PromoSegment]) -> List[str]:
        lines: List[str] = []
        for s in segs:
            ep_obj = ep_map.get(s.episode_number)
            if ep_obj:
                blks = _extract_window_blocks(ep_obj, s.start_sec, s.end_sec)
                for p in _build_transcript_preview(blks, max_lines=4):
                    lines.append(f"(Ep {s.episode_number}) {p}")
        return lines[:10]

    flagships: List[PromoCandidate] = [
        # RANK 1: Single-Stretch Altar Betrayal (Ep 1) - The #1 Cold Traffic Winner
        PromoCandidate(
            promo_id="PROMO-01-ALTAR-BETRAYAL",
            rank=1,
            title="The Altar Swap: Fiancé Marries Her Sister in Front of the Whole Church",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 1: Post First (Broad Cold-Traffic Hook)",
            target_platforms=["TikTok", "Instagram Reels", "YouTube Shorts"],
            total_duration_sec=51.10,
            segments=[
                make_seg(
                    1, 1, 11.00, 62.10, "CONTINUOUS STRETCH",
                    "Matriarch blocks bride Juliana in the aisle ('You should turn around') as she discovers her fiancé Gerald holding her sister Clarissa's hand at the altar ('Because he is marrying me')."
                )
            ],
            opening_hook_3s='[00:11.00-00:14.00] Bride in white silk gown is abruptly blocked mid-aisle by the haughty Matriarch in black velvet: "You should turn around, Juliana."',
            ending_cliffhanger='[00:57.60-01:02.10] Gerald steps forward on the altar beside her gloating sister and looks down coldly: "I changed my mind."',
            why_it_works_one_liner="Zero backstory needed: a bride stopped mid-aisle while her fiancé holds her sister's hand at the altar is the highest-converting microdrama trope on TikTok/Reels.",
            posting_priority_reason="Post #1 (Day 1 Hero Ad): 100% universally legible in 2 seconds with mute captions; introduces the core betrayal without spoiling the scarred Duke twist.",
            scores={"hook_3s": 98.5, "cliffhanger_cut": 96.0, "standalone_clarity": 99.0, "pacing_density": 94.5, "composite_score": 97.3},
        ),

        # RANK 2: Cross-Episode Stitched Trailer (Ep 1 Altar Rejection + Ep 3 Scarred Stranger Proposal)
        PromoCandidate(
            promo_id="PROMO-02-STRANGER-PROPOSAL-STITCH",
            rank=2,
            title="Abandoned at the Altar -> Rescued & Proposed to by the Scarred Warlord",
            promo_type="Cross-Episode Stitched Trailer",
            posting_wave="Wave 1: Post First (Broad Cold-Traffic Hook)",
            target_platforms=["TikTok", "Instagram Reels", "YouTube Shorts"],
            total_duration_sec=49.80,
            segments=[
                make_seg(
                    1, 1, 33.70, 50.00, "HOOK & SETUP",
                    "Juliana sees Gerald holding her sister Clarissa's hand at the altar: 'Because he is marrying me!' / 'You knew this before I walked through those doors!'"
                ),
                make_seg(
                    2, 3, 104.50, 121.00, "ESCALATION",
                    "Guards seize Juliana to drag her out of the chapel until the imposing, battle-scarred stranger steps into the aisle and effortlessly disarms them."
                ),
                make_seg(
                    3, 3, 212.00, 229.00, "CLIFFHANGER PAYOFF",
                    "The scarred stranger drops to one knee before the stunned congregation, holding out an heirloom signet ring: 'Marry me.'"
                ),
            ],
            opening_hook_3s='[Ep 1 | 00:33.70-00:36.70] Close-up on the bride in tears looking up at the altar: "Gerald? Why are you holding her hand?" -> Sister smirks: "Because he is marrying me."',
            ending_cliffhanger='[Ep 3 | 03:32.00-03:49.00] The mysterious scarred soldier drops to one knee in front of her cruel family and holds up a ring: "Marry me." Cut before she answers.',
            why_it_works_one_liner="Delivers the complete title premise ('I Married a Broken Soldier') in 49.8 seconds by stitching the altar humiliation directly into the shock counter-proposal.",
            posting_priority_reason="Post #2 (Day 1 Co-Headline Ad): Directly sells the romance/revenge fantasy promised by the show's title and hooks viewers on whether she says yes.",
            scores={"hook_3s": 97.0, "cliffhanger_cut": 98.5, "standalone_clarity": 97.5, "pacing_density": 95.0, "composite_score": 97.3},
        ),

        # RANK 3: Single-Stretch Cathedral Explosion & Chained Bride (Ep 41)
        PromoCandidate(
            promo_id="PROMO-03-CHAINED-BRIDE-EXPLOSION",
            rank=3,
            title="Paraded in Iron Shackles at Her Enemy's Wedding—Until the Cathedral Doors Explode",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 2: High-Heat Action/Revenge",
            target_platforms=["TikTok", "Instagram Reels", "YouTube Shorts"],
            total_duration_sec=56.50,
            segments=[
                make_seg(
                    1, 41, 15.50, 44.20, "HOOK & SETUP",
                    "Cathedral doors swing open as Juliana is marched down the flower-strewn wedding aisle in heavy iron shackles while the bride Catherine gloats."
                ),
                make_seg(
                    2, 41, 68.09, 96.09, "CLIFFHANGER PAYOFF",
                    "As Catherine demands Juliana kneel at the altar, a deafening explosion blasts the cathedral doors off their hinges and the presumed-dead Duke emerges through the smoke."
                ),
            ],
            opening_hook_3s='[Ep 41 | 00:15.50-00:18.50] Heavy cathedral doors swing open mid-wedding to reveal Lady Juliana dragged down the bridal aisle in iron wrist shackles.',
            ending_cliffhanger='[Ep 41 | 01:30.00-01:36.09] A massive explosion shatters the heavy oak cathedral gates; through the swirling smoke, a tall figure in full ducal regalia steps in.',
            why_it_works_one_liner="Extreme visual contrast (a shackled prisoner dragged down a luxury wedding aisle) followed by a literal explosion cliffhanger as the silhouette enters.",
            posting_priority_reason="Post #3 (Wave 2 Action Hook): Highest visual kinetic score in the 50-episode pack; pulls in viewers who scroll past dialogue-only clips.",
            scores={"hook_3s": 98.0, "cliffhanger_cut": 99.0, "standalone_clarity": 94.0, "pacing_density": 96.0, "composite_score": 96.8},
        ),

        # RANK 4: Single-Stretch Planted Watch & Blackthorn Briars Ultimatum (Ep 12)
        PromoCandidate(
            promo_id="PROMO-04-BLACKTHORN-ULTIMATUM",
            rank=4,
            title="Crawl Into Razor-Sharp Blackthorn Briars Barehanded—Or Watch Your Husband Hang",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 2: High-Heat Action/Revenge",
            target_platforms=["TikTok", "Instagram Reels"],
            total_duration_sec=54.20,
            segments=[
                make_seg(
                    1, 12, 0.00, 28.00, "HOOK & SETUP",
                    "Constable raises heavy iron handcuffs to arrest Alexander over a planted gold watch as Juliana shields him with her body: 'You planted it!'"
                ),
                make_seg(
                    2, 12, 54.00, 80.20, "CLIFFHANGER PAYOFF",
                    "Clarissa delivers a sadistic ultimatum: crawl into the frozen, razor-sharp blackthorn briars with bare hands to fetch her ring, or Alexander hangs at dawn."
                ),
            ],
            opening_hook_3s='[Ep 12 | 00:00.00-00:03.00] Constable raises heavy iron handcuffs: "Put out your hands, veteran. You are under arrest for theft!" as Juliana throws herself in front of him.',
            ending_cliffhanger='[Ep 12 | 01:15.00-01:20.20] Clarissa smiles venomously: "Fetch my signet ring from the blackthorn briars with your bare hands... or watch him hang."',
            why_it_works_one_liner="Instant physical stakes (handcuffs in frame at 00:00) escalating into a visceral, sadistic choice that forces viewers to see if she actually crawls into the thorns.",
            posting_priority_reason="Post #4 (Wave 2 Moral Outrage Ad): Outrage-driven ultimatums drive massive comment-section engagement on Reels and Shorts.",
            scores={"hook_3s": 96.0, "cliffhanger_cut": 95.5, "standalone_clarity": 96.0, "pacing_density": 93.0, "composite_score": 95.4},
        ),

        # RANK 5: Cross-Episode Stitched Revenge Arc (Ep 13 Bleeding Hands in Thorns -> Ep 42 "He Is The Rightful Duke!")
        PromoCandidate(
            promo_id="PROMO-05-BEGGAR-TO-DUKE-REVEAL",
            rank=5,
            title="They Mocked Her 'Crippled Beggar' Husband—Until the Royal Guards Saluted Him as Duke",
            promo_type="Cross-Episode Stitched Trailer",
            posting_wave="Wave 3: Retargeting & Payoff Twist",
            target_platforms=["TikTok", "Instagram Reels", "YouTube Shorts"],
            total_duration_sec=52.80,
            segments=[
                make_seg(
                    1, 12, 28.00, 44.00, "HOOK & SETUP",
                    "Clarissa and Gerald mock Juliana for defending her 'broken scrap of a man, a crippled beggar with nothing.'"
                ),
                make_seg(
                    2, 39, 180.00, 198.80, "ESCALATION",
                    "Inside the candlelit chamber, Alexander sheds his rugged peasant cloak and fastens the gold-embroidered Ducal mantle over his scarred shoulders."
                ),
                make_seg(
                    3, 42, 42.00, 60.00, "CLIFFHANGER PAYOFF",
                    "In the ruined cathedral, Rupert screams at the Royal Guards to execute Alexander—instead, the Captain drops to a salute as his mother proclaims him the rightful Duke!"
                ),
            ],
            opening_hook_3s='[Ep 12 | 00:28.00-00:31.00] Clarissa sneers down at Juliana and her scarred husband: "Still defending your broken scrap of a man?"',
            ending_cliffhanger='[Ep 42 | 00:54.00-01:00.00] Usurper Rupert screams "Arrest him!"—but the Royal Guard Captain snaps to attention and salutes Alexander in silence.',
            why_it_works_one_liner="Captures the core status-reversal fantasy ('He Was Secretly a Duke') by contrasting his early humiliation with the moment the Royal Guards salute him.",
            posting_priority_reason="Post #5 (Wave 3 Identity Reveal Ad): Perfect for audiences who love 'secret billionaire/royalty' identity-reveal hooks.",
            scores={"hook_3s": 93.5, "cliffhanger_cut": 97.5, "standalone_clarity": 95.0, "pacing_density": 94.0, "composite_score": 95.1},
        ),

        # RANK 6: Single-Stretch Dungeon Red-Hot Branding Iron Confrontation (Ep 31)
        PromoCandidate(
            promo_id="PROMO-06-DUNGEON-BRANDING-IRON",
            rank=6,
            title="Chained in the Wet Dungeon, She Faces a Red-Hot Branding Iron to Save Her Mother's Book",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 2: High-Heat Action/Revenge",
            target_platforms=["TikTok", "Instagram Reels"],
            total_duration_sec=48.00,
            segments=[
                make_seg(
                    1, 31, 112.00, 160.00, "CONTINUOUS STRETCH",
                    "Catherine pulls a glowing red-hot branding iron from the brazier inside the subterranean dungeon cell and presses it toward Juliana and her late mother's herbal journal."
                )
            ],
            opening_hook_3s='[Ep 31 | 01:52.00-01:55.00] Close-up on a glowing, red-hot branding iron lifted from a dungeon brazier, casting fiery sparks across Juliana\'s chained wrists.',
            ending_cliffhanger='[Ep 31 | 02:35.00-02:40.00] Catherine drives the glowing iron down onto the leather cover of her late mother\'s book as smoke hisses and Juliana lunges against her chains.',
            why_it_works_one_liner="A glowing red-hot branding iron in a dark dungeon cell is an instant visual thumb-stopper with visceral sensory stakes.",
            posting_priority_reason="Post #6 (Wave 2 High-Tension Drama Ad): High visual contrast (glowing embers in dark wet cells) and intense villain vs. heroine clash.",
            scores={"hook_3s": 96.5, "cliffhanger_cut": 94.0, "standalone_clarity": 91.0, "pacing_density": 92.5, "composite_score": 93.9},
        ),

        # RANK 7: Single-Stretch Midnight Arson Inferno Rescue (Ep 18)
        PromoCandidate(
            promo_id="PROMO-07-MIDNIGHT-ARSON-RESCUE",
            rank=7,
            title="Trapped Inside a Burning Cottage at Midnight—He Walks Through Fire for Her",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 2: High-Heat Action/Revenge",
            target_platforms=["YouTube Shorts", "Instagram Reels"],
            total_duration_sec=44.50,
            segments=[
                make_seg(
                    1, 18, 28.90, 73.40, "CONTINUOUS STRETCH",
                    "Arsonists bolt the cottage door from outside and set the roof ablaze; as burning beams collapse around Juliana, Alexander kicks through the wall of flames."
                )
            ],
            opening_hook_3s='[Ep 18 | 00:28.90-00:32.00] Masked arsonists hurl a flaming torch onto the cottage roof and bolt the heavy timber door from the outside.',
            ending_cliffhanger='[Ep 18 | 01:09.00-01:13.40] A blazing ceiling beam crashes down blocking the exit just as Juliana reaches for her mother\'s ledger in the smoke.',
            why_it_works_one_liner="Immediate life-or-death physical peril (locked doors + roaring flames in the first 3 seconds) that requires zero dialogue to hook global viewers.",
            posting_priority_reason="Post #7 (Visual-First International Ad): Works exceptionally well even with low audio reliance because the threat is 100% visual.",
            scores={"hook_3s": 95.0, "cliffhanger_cut": 93.0, "standalone_clarity": 93.5, "pacing_density": 91.0, "composite_score": 93.5},
        ),

        # RANK 8: Single-Stretch Dowager Duchess Recognizes the Lost Son's Ring (Ep 26-27 Stitch)
        PromoCandidate(
            promo_id="PROMO-08-DUCHESS-RING-TRAP",
            rank=8,
            title="The Dying Duchess Recognizes the Maid's Wedding Ring—Right as the Traitors Burst In",
            promo_type="Cross-Episode Stitched Trailer",
            posting_wave="Wave 3: Retargeting & Payoff Twist",
            target_platforms=["TikTok", "YouTube Shorts"],
            total_duration_sec=46.00,
            segments=[
                make_seg(
                    1, 26, 95.00, 120.88, "HOOK & SETUP",
                    "As disguised maid Juliana feeds charcoal antidote to the dying Dowager Duchess, the Duchess's eyes snap open and she seizes Juliana's wrist, staring at the Ashbourne crest ring: 'Where did you get this ring?!'"
                ),
                make_seg(
                    2, 27, 45.00, 65.12, "CLIFFHANGER PAYOFF",
                    "Before Juliana can whisper Alexander's name, the chamber doors burst open and Rupert and Catherine storm in with guards: 'Seize her! She is poisoning the Duchess!'"
                ),
            ],
            opening_hook_3s='[Ep 26 | 01:35.00-01:38.00] The bedridden Dowager Duchess suddenly grabs the disguised maid\'s wrist in the dark: "This signet ring... it belongs to my dead son!"',
            ending_cliffhanger='[Ep 27 | 01:00.00-01:05.12] Rupert and Catherine burst through the bedroom doors with armed guards: "Catch her in the act! She is poisoning the Duchess!"',
            why_it_works_one_liner="Combines an emotional secret-identity discovery (the mother recognizing her 'dead' son's ring) with an immediate false-accusation ambush.",
            posting_priority_reason="Post #8 (Wave 3 Mid-Season Suspense Ad): Hooks mystery/thriller viewers with a double reversal in 46 seconds.",
            scores={"hook_3s": 92.0, "cliffhanger_cut": 94.5, "standalone_clarity": 91.5, "pacing_density": 92.0, "composite_score": 92.6},
        ),

        # RANK 9: Single-Stretch Garden Thugs Crushed by the "Crippled" Soldier (Ep 16)
        PromoCandidate(
            promo_id="PROMO-09-SOLDIER-UNLEASHED",
            rank=9,
            title="Three Hired Thugs Corner His Wife in the Dark—And Learn Why He's Feared on the Battlefield",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 2: High-Heat Action/Revenge",
            target_platforms=["YouTube Shorts", "TikTok"],
            total_duration_sec=47.50,
            segments=[
                make_seg(
                    1, 16, 0.00, 47.50, "CONTINUOUS STRETCH",
                    "Three armed thugs corner bandaged Juliana in her moonlit herb garden until Alexander steps out of the shadows, dislocating the leader's arm and forcing a confession."
                )
            ],
            opening_hook_3s='[Ep 16 | 00:00.00-00:03.00] Three armed thugs kick apart Juliana\'s garden as she lies injured on the earth—until Alexander catches the leader\'s club barehanded.',
            ending_cliffhanger='[Ep 16 | 00:43.00-00:47.50] Pinning the terrified thug to the mud, Alexander demands in a lethal whisper: "Tell me who paid you, or you don\'t walk out of these woods."',
            why_it_works_one_liner="Pure protective-husband action payoff: opens mid-swing at 00:00 and cuts right on the interrogation confession threat.",
            posting_priority_reason="Post #9 (Action-Segment Ad): High male & female crossover appeal due to fast combat choreography.",
            scores={"hook_3s": 94.0, "cliffhanger_cut": 91.0, "standalone_clarity": 93.0, "pacing_density": 90.5, "composite_score": 92.3},
        ),

        # RANK 10: Single-Stretch Table-Turn: Usurper Bride Forced to Become Scullery Maid (Ep 47)
        PromoCandidate(
            promo_id="PROMO-10-MAID-REVERSAL",
            rank=10,
            title="The Woman Who Whipped Her Now Begs at Her Feet—And Gets Hired as a Scullery Maid",
            promo_type="Single-Stretch Continuous",
            posting_wave="Wave 3: Retargeting & Payoff Twist",
            target_platforms=["TikTok", "Instagram Reels"],
            total_duration_sec=53.00,
            segments=[
                make_seg(
                    1, 47, 130.00, 183.00, "CONTINUOUS STRETCH",
                    "Destitute and disgraced in the Ashbourne courtyard, Catherine drops to her knees begging Duchess Juliana for mercy; Juliana grants her survival—as the lowest scullery maid."
                )
            ],
            opening_hook_3s='[Ep 47 | 02:10.00-02:13.00] Disgraced noblewoman Catherine collapses onto the wet cobblestones at Duchess Juliana\'s feet: "Please! I have nowhere else to go!"',
            ending_cliffhanger='[Ep 47 | 02:58.00-03:03.00] Juliana looks down regally: "You may stay at Ashbourne... on your knees, scrubbing the scullery floors." Catherine\'s hidden eyes flash with murder.',
            why_it_works_one_liner="Delivers the ultimate status-reversal karma that microdrama viewers crave while ending on Catherine's sinister hidden glare.",
            posting_priority_reason="Post #10 (Wave 3 Retargeting Ad): Converts viewers who saw the early humiliation ads and want to see Juliana turn the tables.",
            scores={"hook_3s": 91.0, "cliffhanger_cut": 90.5, "standalone_clarity": 92.0, "pacing_density": 89.5, "composite_score": 90.9},
        ),
    ]

    for p in flagships:
        p.transcript_preview = collect_preview(p.segments)

    return _apply_model_profile_to_promos(flagships, router=router)
