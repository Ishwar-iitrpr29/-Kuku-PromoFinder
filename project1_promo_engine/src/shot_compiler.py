"""
Module 3D: ShotForge Character-Locked AI Video & TTS Asset Compiler.
Compiles any selected promo clip into a production-ready JSON/CSV Shot Sheet
with stateful character appearance prompts (injuries, gown state, lighting)
and canonical ElevenLabs/TTS voice routing IDs.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from .parser import EpisodeScript, format_mmss
from .promo_finder import PromoCandidate
from .llm_router import UniversalLLMRouter


VOICE_PROFILE_REGISTRY = {
    "LADY JULIANA ASHCROFT": {
        "tts_voice_id": "kuku_en_us_juliana_v2",
        "vocal_timbre": "Young aristocratic soprano, resilient, emotional dynamic range",
        "visual_lora_token": "<lora:juliana_ashcroft_regency:0.85>",
    },
    "DUKE ALEXANDER ASHBOURNE": {
        "tts_voice_id": "kuku_en_us_alexander_baritone_v2",
        "vocal_timbre": "Deep gravelly baritone, commanding, measured intensity",
        "visual_lora_token": "<lora:alexander_scarred_duke:0.90>",
    },
    "GERALD HARRINGTON (EX-FIANCE)": {
        "tts_voice_id": "kuku_en_us_gerald_tenor_v1",
        "vocal_timbre": "Haughty aristocratic tenor, cold, dismissive",
        "visual_lora_token": "<lora:gerald_harrington_groom:0.85>",
    },
    "LADY CLARISSA (SISTER)": {
        "tts_voice_id": "kuku_en_us_clarissa_alto_v1",
        "vocal_timbre": "Sharp, mocking, syrupy aristocratic alto",
        "visual_lora_token": "<lora:clarissa_gold_gown:0.85>",
    },
    "LADY ASHCROFT (JULIANA'S STEPMOTHER)": {
        "tts_voice_id": "kuku_en_us_matriarch_contralto_v1",
        "vocal_timbre": "Imperious elder contralto, stern, icy",
        "visual_lora_token": "<lora:lady_ashcroft_matriarch:0.85>",
    },
    "LORD ASHCROFT (JULIANA'S FATHER)": {
        "tts_voice_id": "kuku_en_us_lord_ashcroft_elder_v1",
        "vocal_timbre": "Weathered elder bass, severe, cold",
        "visual_lora_token": "<lora:lord_ashcroft_cane:0.85>",
    },
    "LADY CATHERINE (USURPER BRIDE)": {
        "tts_voice_id": "kuku_en_us_catherine_villain_v2",
        "vocal_timbre": "Cruel, regal, volatile mezzo-soprano",
        "visual_lora_token": "<lora:lady_catherine_ashbourne:0.85>",
    },
    "LORD RUPERT ASHBOURNE (USURPER COUSIN)": {
        "tts_voice_id": "kuku_en_us_rupert_usurper_v1",
        "vocal_timbre": "Arrogant, defensive, scheming noble tenor",
        "visual_lora_token": "<lora:rupert_blond_usurper:0.85>",
    },
    "DOWAGER DUCHESS ASHBOURNE (ALEXANDER'S MOTHER)": {
        "tts_voice_id": "kuku_en_us_dowager_duchess_v1",
        "vocal_timbre": "Frail yet fiercely dignified elder matriarch",
        "visual_lora_token": "<lora:dowager_duchess_ashbourne:0.85>",
    },
}


def _get_episode_visual_state(episode_number: int) -> str:
    if episode_number <= 5:
        return "Juliana in vintage white silk bridal gown + heirloom pendant; Cathedral candlelit interior."
    elif episode_number <= 13:
        return "Juliana in rustic linen cottage dress; hands torn/bleeding in Ep 13 blackthorn garden."
    elif episode_number <= 17:
        return "Juliana with white linen bandages wrapped around both hands; Alexander in rugged wool greatcoat."
    elif episode_number <= 21:
        return "Soot-stained cloaks post-cottage fire; dim lantern-lit barn & village setting."
    elif episode_number <= 28:
        return "Juliana disguised in drab grey scullery maid uniform + apron; shoulder laceration from riding crop (Ep 24+)."
    elif episode_number <= 38:
        return "Juliana in torn coarse prisoner smock, iron wrist shackles, damp subterranean stone dungeon."
    else:
        return "Cathedral wedding climax: Juliana in prisoner smock & iron chains (freed in Ep 45); Alexander in full crimson-and-gold Ducal regalia."


@dataclass
class CompiledShot:
    shot_index: int
    episode_number: int
    timestamp_window: str
    duration_sec: float
    block_type: str
    canonical_speaker: str
    tts_voice_id: str
    emotion_direction: str
    persistent_wardrobe_state: str
    visual_keyframe_prompt: str
    dialogue_or_action: str


def compile_promo_shot_pack(
    promo: PromoCandidate,
    episodes: List[EpisodeScript],
    router: Optional[UniversalLLMRouter] = None,
) -> Dict:
    ep_map = {e.episode_number: e for e in episodes}
    shots: List[CompiledShot] = []
    idx = 1

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
        camera_style_prefix = "Vertical 9:16 High-Intensity Indian Royal Microdrama, dramatic rim-lighting, fast push-in lens,"
        default_emotion = "High-Voltage Melodrama & Confrontation"
    elif is_urban_model:
        camera_style_prefix = "Vertical 9:16 Modern Fast-Cut Reels Aesthetic, crisp shallow depth-of-field, dynamic handheld framing,"
        default_emotion = "Punchy Conversational Realism"
    else:
        camera_style_prefix = "Vertical 9:16 cinematic Regency drama frame, chiaroscuro candlelight,"
        default_emotion = "Cinematic Regency Tension"

    for seg in promo.segments:
        ep = ep_map.get(seg.episode_number)
        if not ep:
            continue
        state_desc = _get_episode_visual_state(ep.episode_number)
        for blk in ep.blocks:
            if blk.end_sec <= seg.start_sec + 0.05 or blk.start_sec >= seg.end_sec - 0.05:
                continue

            prof = VOICE_PROFILE_REGISTRY.get(blk.canonical_speaker, {
                "tts_voice_id": "kuku_en_us_ensemble_default",
                "vocal_timbre": "Standard dramatic delivery",
                "visual_lora_token": "<lora:regency_ensemble:0.75>",
            })

            # Auto-sanitize any historical anachronism in the visual prompt
            clean_text = blk.text.replace("with a numbered identification badge", "in iron wrist shackles")
            clean_text = clean_text.replace("trench coat", "military wool greatcoat")

            visual_prompt = (
                f"{camera_style_prefix} {state_desc} "
                f"{prof['visual_lora_token']} Shot: {clean_text}"
            )

            shots.append(
                CompiledShot(
                    shot_index=idx,
                    episode_number=ep.episode_number,
                    timestamp_window=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                    duration_sec=round(max(0.5, blk.end_sec - blk.start_sec), 2),
                    block_type=blk.block_type,
                    canonical_speaker=blk.canonical_speaker if blk.block_type == "DIALOGUE" else "N/A",
                    tts_voice_id=prof["tts_voice_id"] if blk.block_type == "DIALOGUE" else "SFX_OR_SILENT",
                    emotion_direction=", ".join(blk.emotion_cues) if blk.emotion_cues else default_emotion,
                    persistent_wardrobe_state=state_desc,
                    visual_keyframe_prompt=visual_prompt,
                    dialogue_or_action=clean_text,
                )
            )
            idx += 1

    return {
        "promo_id": promo.promo_id,
        "promo_title": promo.title,
        "total_duration_sec": promo.total_duration_sec,
        "shot_count": len(shots),
        "tts_voices_used": sorted(list({s.tts_voice_id for s in shots if s.tts_voice_id != "SFX_OR_SILENT"})),
        "shots": [asdict(s) for s in shots],
    }
