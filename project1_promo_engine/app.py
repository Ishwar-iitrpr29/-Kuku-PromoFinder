"""
Kuku PromoFinder — Clean, Focused Streamlit App (Project 1).
Run locally:
    python -m streamlit run app.py
"""

from __future__ import annotations
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure local src directory is on sys.path
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import src.llm_router
import src.parser
import src.entity_resolver
import src.pacing_analyzer
import src.promo_finder
import src.quality_auditor
import src.localizer
import src.shot_compiler

for _mod in (
    src.llm_router,
    src.parser,
    src.entity_resolver,
    src.pacing_analyzer,
    src.promo_finder,
    src.quality_auditor,
    src.localizer,
    src.shot_compiler,
):
    importlib.reload(_mod)

from src.llm_router import (
    LLMConfig,
    UniversalLLMRouter,
    PROVIDER_DEFAULTS,
    PROVIDER_MODELS,
    get_available_providers_from_env,
)
from src.parser import parse_all_episodes, parse_raw_script_text
from src.entity_resolver import run_entity_and_continuity_audit
from src.pacing_analyzer import analyze_pacing_and_overlaps
from src.promo_finder import (
    find_best_single_stretch_per_episode,
    build_curated_flagship_and_stitched_promos,
)
from src.quality_auditor import run_quality_and_failure_audit
from src.localizer import localize_promo_isochronous
from src.shot_compiler import compile_promo_shot_pack


st.set_page_config(
    page_title="Kuku PromoFinder",
    page_icon="🎬",
    layout="wide",
)

_local_episodes_dir = Path(__file__).resolve().parent / "data" / "episodes"
_parent_episodes_dir = (
    Path(__file__).resolve().parent.parent
    / "I Married a Broken Soldier. He was a Duke-scripts (2)"
    / "I Married a Broken Soldier. He was a Duke"
)
DEFAULT_SCRIPTS_DIR = _local_episodes_dir if _local_episodes_dir.is_dir() else _parent_episodes_dir

CUSTOM_SCRIPT_PRESETS = {
    "CT01: Valid Unformatted Plain-Text Scene (~28s Altar Confrontation)": (
        "[CAMERA] Extreme close-up on Juliana's trembling hands in a white bridal gown as the heavy cathedral doors slam shut.\n"
        "Lady Ashcroft (cold, commanding): You should turn around, Juliana. Stop embarrassing this family.\n"
        "Juliana (shocked, breathless): Gerald? Why are you standing at the altar holding Clarissa's hand?\n"
        "Clarissa (gloating, smug): Because he is marrying me! Did you really think a disgraced daughter would wear the Harrington ring?\n"
        "Gerald (callous): I changed my mind, Juliana. Step aside before the guards drag you into the mud.\n"
        "[ACTION] Armed estate guards seize Juliana's arms as she struggles in her torn silk bridal veil.\n"
        "Alexander (deep, dangerous): Take your hands off her, or nobody walks out of this church alive!\n"
        "[SOUND FX] Heavy oak doors shatter open as thunder echoes through the nave; cut to black."
    ),
    "CT02: Too-Short Scene (<20s Minimum Ad Rule)": (
        "[ACTION] Juliana walks into the candlelit library holding an old leather book.\n"
        "Juliana (whispering): So this was my mother's secret ledger all along.\n"
        "Alexander (calm): Put it away before someone hears us in the corridor!"
    ),
    "CT03: Off-Topic Non-Screenplay Input (Cooking Recipe)": (
        "How to cook Hyderabadi Chicken Biryani: First marinate 500g chicken with yogurt, ginger garlic paste, "
        "red chili powder, turmeric, and garam masala for 2 hours. Boil basmati rice with whole spices until 70% done, "
        "then layer over the chicken and cook on dum for 25 minutes."
    ),
    "CT04: Random Gibberish / Keysmash": "asdfghjkl qwertyuiop 1234567890 zxcvbnm ??? !!!",
}


@st.cache_data(show_spinner=False)
def load_and_analyze_pack(folder_str: str, cache_v: int = 3):
    episodes = parse_all_episodes(folder_str)
    entity_audit = run_entity_and_continuity_audit(episodes)
    pacing_audit = analyze_pacing_and_overlaps(episodes)
    per_ep_promos = find_best_single_stretch_per_episode(episodes)
    quality_audit = run_quality_and_failure_audit(episodes)
    return episodes, entity_audit, pacing_audit, per_ep_promos, quality_audit


def analyze_custom_script_text(raw_text: str):
    ep = parse_raw_script_text(raw_text, filename="custom_episode_1.txt")
    episodes = [ep]
    entity_audit = run_entity_and_continuity_audit(episodes)
    pacing_audit = analyze_pacing_and_overlaps(episodes)
    per_ep_promos = find_best_single_stretch_per_episode(episodes)
    quality_audit = run_quality_and_failure_audit(episodes)
    return episodes, entity_audit, pacing_audit, per_ep_promos, quality_audit


def main():
    st.markdown("## 🎬 Kuku PromoFinder")
    st.caption("Automatically finds, ranks, and localizes the highest-converting 20–60s ad cuts across all 50 episodes.")

    # Minimal Sidebar
    with st.sidebar:
        st.markdown("### 📂 Scripts Input")
        input_mode = st.radio(
            "Source",
            options=[
                "Default 50-Episode Pack",
                "Upload .txt Scripts",
                "✍️ Paste / Test Custom Script",
            ],
            index=0,
            label_visibility="collapsed",
        )

        active_folder = str(DEFAULT_SCRIPTS_DIR)
        custom_raw_text = ""

        if input_mode == "Upload .txt Scripts":
            uploaded = st.file_uploader("Upload episode .txt files", type=["txt"], accept_multiple_files=True)
            if uploaded:
                tmp_dir = Path(tempfile.mkdtemp(prefix="kuku_upload_"))
                for uf in uploaded:
                    (tmp_dir / uf.name).write_bytes(uf.getvalue())
                active_folder = str(tmp_dir)
                st.success(f"Loaded {len(uploaded)} episodes")

        elif input_mode == "✍️ Paste / Test Custom Script":
            preset_choice = st.selectbox(
                "Quick Test Scenario",
                options=list(CUSTOM_SCRIPT_PRESETS.keys()),
                index=0,
            )
            custom_raw_text = st.text_area(
                "Paste or type screenplay / test text:",
                value=CUSTOM_SCRIPT_PRESETS[preset_choice],
                height=190,
                help="Supports both timestamped Kuku scripts (`**SPEAKER**:` + `Start: (MM:SS)`) and plain-text scenes (`Speaker: Dialogue` or `[ACTION] ...`). Also tests gibberish and off-topic rejection.",
            )

        with st.expander("⚙️ AI Model Settings (Optional)", expanded=False):
            available_routers, env_configured = get_available_providers_from_env()
            provider = st.selectbox(
                "AI Router (from .env)",
                options=available_routers,
                index=0,
            )
            available_models = PROVIDER_MODELS.get(provider, PROVIDER_MODELS["gemini"])
            selected_model = st.selectbox(
                "Select Model",
                options=available_models,
                index=0,
            )
            selected_profile = st.selectbox(
                "Model Engine Profile",
                options=[
                    "Standard Hybrid Promo Engine",
                    "Bollywood Dramatic Microdrama",
                    "Conversational Urban Hinglish",
                ],
                index=0,
            )
            model_name = f"{selected_model} ({selected_profile})"
            if env_configured:
                st.caption(f"🔒 Active `.env` Routers: `{', '.join(env_configured)}` · Guardrails Active")
            else:
                st.caption("🔒 Add `GEMINI_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY` in `.env` to unlock more routers · Guardrails Active")

    router = UniversalLLMRouter(LLMConfig(provider=provider, model=model_name))

    try:
        if input_mode == "✍️ Paste / Test Custom Script":
            episodes, entity_audit, pacing_audit, per_ep_promos, quality_audit = analyze_custom_script_text(custom_raw_text)
        else:
            episodes, entity_audit, pacing_audit, per_ep_promos, quality_audit = load_and_analyze_pack(active_folder)
        top_promos = build_curated_flagship_and_stitched_promos(episodes, router=router)
    except Exception as exc:
        st.error(f"Could not read scripts: {exc}")
        return

    # Check if the input was flagged as Gibberish, Off-Topic Non-Screenplay, or Too Short
    first_ep_status = episodes[0].input_status if len(episodes) == 1 else "VALID_TIMESTAMPED_SCRIPT"
    first_ep_msg = episodes[0].status_message if len(episodes) == 1 else ""

    if not top_promos or first_ep_status in ("GIBBERISH_INPUT", "NOT_A_SCREENPLAY"):
        st.error(
            f"### 🚫 Unrecognized or Non-Screenplay Input (`{first_ep_status}`)\n\n"
            f"**Why Promo Generation Was Blocked:** {first_ep_msg or 'No valid dramatic dialogue or camera/action blocks were found in the input.'}\n\n"
            "**How to Fix:** Paste or upload a screenplay scene containing character dialogue (e.g., `Juliana (furious): Why are you holding her hand?` or `**JULIANA**:` with `Start: (MM:SS) | End: (MM:SS)`) and dramatic action beats (`[ACTION] ...`)."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Episodes Analyzed", "0 (Rejected)", first_ep_status)
        m2.metric("Top Flagship Promos", "0", "Blocked by Input Gate")
        m3.metric("Per-Episode Cuts", "0", "Need >= 20s scene")
        m4.metric("Script Fixes Applied", "0", "N/A")
        return

    if first_ep_status == "SCRIPT_TOO_SHORT_FOR_PROMO":
        st.warning(
            f"### ⏳ Scene Under 20-Second Standalone Ad Minimum (`{episodes[0].total_duration_sec:.1f}s`)\n\n"
            f"{first_ep_msg} Below is the preview analysis, synthesized timing, and Hindi localization so you can expand the scene."
        )
    elif first_ep_status == "SYNTHESIZED_PLAIN_SCRIPT":
        st.info(
            f"### ✨ Plain-Text Screenplay Auto-Timed (`{episodes[0].total_duration_sec:.1f}s` across `{len(episodes[0].blocks)}` beats)\n\n"
            f"{first_ep_msg}"
        )

    # Compact 4-Metric Summary Bar
    total_min = sum(e.total_duration_sec for e in episodes) / 60.0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Episodes Analyzed", f"{len(episodes)}", f"{total_min:.1f} mins total")
    m2.metric("Top Flagship Promos", f"{len(top_promos)}", "Single + Multi-Ep")
    m3.metric("Per-Episode Cuts", f"{len(per_ep_promos)}", "20–60s each")
    m4.metric("Script Fixes Applied", f"{entity_audit['total_anomalies_found']}", "Names & Speakers")

    tab_ranked, tab_episodes, tab_quality = st.tabs([
        "🏆 Top 10 Ranked Promos",
        "📺 Episode-by-Episode Cuts (1–50)",
        "🛡️ Quality Check & Script Fixes",
    ])

    # =========================================================================
    # TAB 1: TOP 10 RANKED PROMOS (Clean Leaderboard + Single Promo Inspector)
    # =========================================================================
    with tab_ranked:
        leaderboard_rows = [
            {
                "Rank": f"#{p.rank}",
                "Promo Title": p.title,
                "Format": "Single Scene" if "Single" in p.promo_type else "Multi-Ep Trailer",
                "Duration": f"{p.total_duration_sec:.0f}s",
                "Score": f"{p.scores['composite_score']}/100",
                "Campaign Wave": p.posting_wave.split("(")[0].strip(),
            }
            for p in top_promos
        ]
        st.dataframe(pd.DataFrame(leaderboard_rows), width="stretch", hide_index=True)

        st.markdown("#### 🔍 Inspect Promo Cut")
        promo_options = [f"Rank #{p.rank}: {p.title} ({p.total_duration_sec:.0f}s)" for p in top_promos]
        selected_label = st.selectbox("Choose a ranked promo to view details:", options=promo_options, index=0, label_visibility="collapsed")
        selected_idx = promo_options.index(selected_label) if selected_label in promo_options else 0
        p = top_promos[selected_idx]

        with st.container(border=True):
            top_l, top_r = st.columns([3, 1])
            with top_l:
                st.markdown(f"### #{p.rank} · {p.title}")
                st.caption(
                    f"**Format:** {p.promo_type}  |  **Duration:** {p.total_duration_sec:.1f}s  |  "
                    f"**Wave:** {p.posting_wave}  |  **Engine:** `{router.config.model}`"
                )
            with top_r:
                st.metric("Promo Score", f"{p.scores['composite_score']}/100", f"Hook: {p.scores['hook_3s']} · Cliff: {p.scores['cliffhanger_cut']}")

            c_hook, c_cliff = st.columns(2)
            with c_hook:
                st.info(f"**⚡ First 3 Seconds (Hook)**\n\n{p.opening_hook_3s}")
            with c_cliff:
                st.warning(f"**🧨 Final Cut (Cliffhanger)**\n\n{p.ending_cliffhanger}")

            st.success(f"**💡 Why This Cut Converts:** {p.why_it_works_one_liner}")

            # Clean 4-subtab view inside the single promo card
            sub_cuts, sub_script, sub_hindi, sub_shots = st.tabs([
                "⏱️ Exact Timestamps",
                "📜 Script Lines",
                "🇮🇳 Hindi / Hinglish Dub",
                "🎥 Video Shot Prompts",
            ])

            with sub_cuts:
                seg_rows = [
                    {
                        "Step": s.segment_order,
                        "Episode": f"Ep {s.episode_number}",
                        "Start": s.start_fmt,
                        "End": s.end_fmt,
                        "Duration": f"{s.duration_sec:.1f}s",
                        "Beat Role": s.segment_role,
                        "Scene Action / Line": s.key_action_or_line,
                    }
                    for s in p.segments
                ]
                st.dataframe(pd.DataFrame(seg_rows), width="stretch", hide_index=True)

            with sub_script:
                if p.transcript_preview:
                    st.code("\n".join(p.transcript_preview), language="text")
                else:
                    st.caption("No transcript preview available.")

            with sub_hindi:
                loc_data = localize_promo_isochronous(p, episodes, router=router)
                st.caption(
                    f"Timestamp-locked lines generated via `{loc_data.get('localization_engine', router.config.model)}` "
                    "so Hindi dubbing fits the exact English video clip duration (`<= 4.6 syllables/sec`)."
                )
                clean_loc = [
                    {
                        "Ep & Time": f"Ep {r['episode_number']} ({r['timestamp_window']})",
                        "Speaker": r["canonical_speaker"],
                        "Original English": r["original_english"],
                        "Hinglish Dub": r["isochronous_hinglish_dub"],
                        "Indian Royal Adaptation": r["indian_royal_transcreation"],
                        "Timing Fit": r["timing_fit_status"],
                    }
                    for r in loc_data["timed_lines"]
                ]
                st.dataframe(pd.DataFrame(clean_loc), width="stretch", hide_index=True)

            with sub_shots:
                shot_data = compile_promo_shot_pack(p, episodes, router=router)
                st.caption("Character-consistent visual prompts and voice IDs for each beat of this promo.")
                clean_shots = [
                    {
                        "Shot": s["shot_index"],
                        "Episode": f"Ep {s['episode_number']}",
                        "Time": s["timestamp_window"],
                        "Speaker": s["canonical_speaker"],
                        "Voice ID": s["tts_voice_id"],
                        "Emotion": s["emotion_direction"],
                        "Line / Action": s["dialogue_or_action"],
                        "Visual Keyframe Prompt": s["visual_keyframe_prompt"],
                    }
                    for s in shot_data["shots"]
                ]
                st.dataframe(pd.DataFrame(clean_shots), width="stretch", hide_index=True)

    # =========================================================================
    # TAB 2: EPISODE-BY-EPISODE CUTS (1-50)
    # =========================================================================
    with tab_episodes:
        col_sel, col_info = st.columns([1, 3])
        with col_sel:
            ep_num = st.number_input("Select Episode (1–50)", min_value=1, max_value=max(1, len(episodes)), value=1, step=1)
        ep_promo = next((x for x in per_ep_promos if x.segments[0].episode_number == ep_num), per_ep_promos[0])
        seg = ep_promo.segments[0]
        curve = next((c for c in pacing_audit["episode_curves"] if c["episode_number"] == ep_num), pacing_audit["episode_curves"][0])

        with col_info:
            st.markdown(
                f"#### Episode {ep_num} Best Cut: `{seg.start_fmt} – {seg.end_fmt}` ({ep_promo.total_duration_sec:.0f}s) · **Score: {ep_promo.scores['composite_score']}/100**"
            )
            st.caption(f"**Why picked:** {ep_promo.why_it_works_one_liner}")

        # Plot tension curve with the chosen promo window shaded in green
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["bucket_seconds"],
                y=curve["tension_values"],
                mode="lines+markers",
                fill="tozeroy",
                name="Scene Tension",
                line=dict(color="#00a884", width=2.5),
            )
        )
        fig.add_vrect(
            x0=seg.start_sec,
            x1=seg.end_sec,
            fillcolor="#00a884",
            opacity=0.18,
            line_width=1,
            annotation_text=f"Selected Promo ({seg.start_fmt}–{seg.end_fmt})",
            annotation_position="top left",
        )
        fig.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Episode Timeline (Seconds)",
            yaxis_title="Tension Score",
        )
        st.plotly_chart(fig, width="stretch")

        c_ep1, c_ep2 = st.columns(2)
        c_ep1.info(f"**⚡ Opening Hook ({seg.start_fmt}):** {ep_promo.opening_hook_3s}")
        c_ep2.warning(f"**🧨 Cliffhanger Cut ({seg.end_fmt}):** {ep_promo.ending_cliffhanger}")

        with st.expander("📋 View Full Episode Promo Table"):
            all_ep_rows = [
                {
                    "Episode": f"Ep {x.segments[0].episode_number}",
                    "Window": f"{x.segments[0].start_fmt} – {x.segments[0].end_fmt}",
                    "Duration": f"{x.total_duration_sec:.0f}s",
                    "Score": x.scores["composite_score"],
                    "Opening Hook": x.opening_hook_3s,
                    "Cliffhanger Cut": x.ending_cliffhanger,
                }
                for x in per_ep_promos
            ]
            st.dataframe(pd.DataFrame(all_ep_rows), width="stretch", hide_index=True)

    # =========================================================================
    # TAB 3: QUALITY CHECK & SCRIPT FIXES
    # =========================================================================
    with tab_quality:
        st.markdown("#### 1. Where Naive AI Picks Badly (5 Blocked Traps)")
        st.caption("Examples of high-emotion moments that make terrible ads — and the rules we added to filter them out.")

        bad_rows = [
            {
                "Trap": fp["failure_archetype"],
                "Episode & Time": f"{fp['episodes_involved']} ({fp['naive_timestamps']})",
                "Why Naive AI Liked It": fp["why_naive_model_picked_it"],
                "Why It Fails as an Ad": fp["why_it_fails_as_a_real_promo"],
                "Fix Applied": fp["guardrail_added_to_fix_it"],
            }
            for fp in quality_audit["documented_failure_cases"]
        ]
        st.dataframe(pd.DataFrame(bad_rows), width="stretch", hide_index=True)

        st.divider()
        st.markdown("#### 2. Script Name & Speaker Disambiguation (`LoreLock`)")
        st.caption("Because the 50 episode scripts were transcribed from video in batches, character labels and spoken names drifted.")

        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                st.markdown("##### 🏰 Family Name Disambiguation")
                st.markdown(
                    "- **`House Ashcroft` (Kept Separate):** Verified as Juliana's birth family (`Lord & Lady Ashcroft`, Ep 1–9).\n"
                    "- **`House Ashbourne` (8 Audio Typos Fixed):** Alexander's Dukedom was mis-transcribed from audio as "
                    "`Ashwin` (Ep 39), `Ashburgh` (Ep 40–41), `Ashburn` (Ep 42–43), `Ashburton` (Ep 45–46), `Ashbridge` (Ep 37), `Ashwood` (Ep 36), `Ashburg` (Ep 25–27), and `Ashbury` (Ep 20–23)."
                )
        with col_b:
            with st.container(border=True):
                st.markdown("##### 🎭 Speaker Tag Collision (`BLOND NOBLEMAN`)")
                st.markdown(
                    "- **53 lines** across the show use the visual label `BLOND NOBLEMAN` for **two different characters**:\n"
                    "  1. **Episodes 2 & 9:** Juliana's father (`LORD ASHCROFT`) at the wedding.\n"
                    "  2. **Episodes 27–44:** Alexander's scheming cousin (`LORD RUPERT ASHBOURNE`).\n"
                    "- Automatically separated by episode context before stitching trailers."
                )

        with st.expander(f"🔍 View All {entity_audit['total_anomalies_found']} Detected Script & Continuity Anomalies"):
            if entity_audit["anomalies"]:
                st.dataframe(pd.DataFrame(entity_audit["anomalies"]), width="stretch", hide_index=True)
            else:
                st.caption("No ASR or speaker anomalies detected in the current script input.")

        with st.expander(
            f"📉 CliffPulse Dead-Air Drop-Off Zones ({pacing_audit['dead_air_zones_count']}) & "
            f"Cross-Episode Recap Overlaps ({pacing_audit['recap_overlaps_count']})"
        ):
            if pacing_audit["dead_air_zones"]:
                st.markdown("**Dead-Air Zones (>20s low action/conflict — auto-excluded from promo windows):**")
                st.dataframe(pd.DataFrame(pacing_audit["dead_air_zones"]), width="stretch", hide_index=True)
            if pacing_audit["recap_overlaps"]:
                st.markdown("**Cross-Episode Recap Overlaps (trimmed during multi-episode trailer stitching):**")
                st.dataframe(pd.DataFrame(pacing_audit["recap_overlaps"]), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
