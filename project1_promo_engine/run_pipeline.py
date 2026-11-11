"""
CLI Batch Runner for Kuku PromoStudio (Project 1).
Runs on any folder of microdrama screenplays (defaults to the 50-episode input pack)
and writes complete, unedited outputs to `raw_outputs/`.

Usage:
  python run_pipeline.py
  python run_pipeline.py --input "path/to/new/scripts" --provider gemini --api-key YOUR_KEY
"""

from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.llm_router import LLMConfig, UniversalLLMRouter
from src.parser import parse_all_episodes, EpisodeScript
from src.entity_resolver import run_entity_and_continuity_audit
from src.pacing_analyzer import analyze_pacing_and_overlaps
from src.promo_finder import (
    find_best_single_stretch_per_episode,
    build_curated_flagship_and_stitched_promos,
    PromoCandidate,
)
from src.quality_auditor import run_quality_and_failure_audit
from src.localizer import localize_promo_isochronous
from src.shot_compiler import compile_promo_shot_pack


DEFAULT_INPUT_REL = Path(__file__).resolve().parent.parent / "I Married a Broken Soldier. He was a Duke-scripts (2)" / "I Married a Broken Soldier. He was a Duke"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "raw_outputs"


def _write_markdown_report(
    out_dir: Path,
    episodes: List[EpisodeScript],
    top_promos: List[PromoCandidate],
    per_ep_promos: List[PromoCandidate],
    quality_audit: dict,
    entity_audit: dict,
    pacing_audit: dict,
    elapsed_sec: float,
    router: UniversalLLMRouter,
) -> None:
    md_lines = [
        "# Kuku PromoStudio — Full 50-Episode Promo & Script Intelligence Report",
        "",
        f"- **Show Analyzed:** *I Married a Broken Soldier. He was a Duke* ({len(episodes)} Episodes)",
        f"- **Total Show Runtime Parsed:** `{sum(e.total_duration_sec for e in episodes) / 60.0:.1f} minutes` (`{sum(len(e.blocks) for e in episodes)}` timestamped camera/action/dialogue beats)",
        f"- **Execution Time:** `{elapsed_sec:.2f} seconds`",
        f"- **Model / Engine Configuration:** `{router.config.provider}` (`{router.config.model}`)",
        "",
        "---",
        "",
        "## Part 1: Top 10 Ranked Flagship Promos (Single-Stretch & Multi-Episode Stitched)",
        "",
        "| Rank | Promo ID | Type |Posting Wave | Total Duration | Episode(s) & Exact Segment Timestamps |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for p in top_promos:
        seg_str = " + ".join(
            f"Ep {s.episode_number} (`{s.start_fmt}-{s.end_fmt}`)" for s in p.segments
        )
        md_lines.append(
            f"| **#{p.rank}** | `{p.promo_id}` | {p.promo_type} | {p.posting_wave} | `{p.total_duration_sec:.1f}s` | {seg_str} |"
        )

    md_lines.extend(["", "### Detailed Breakdown of Each Ranked Flagship Promo", ""])
    for p in top_promos:
        md_lines.append(f"#### Rank #{p.rank}: {p.title} (`{p.promo_id}`)")
        md_lines.append(f"- **Type:** {p.promo_type} | **Total Duration:** `{p.total_duration_sec:.2f}s` | **Wave:** {p.posting_wave}")
        md_lines.append(f"- **Scores:** Composite `{p.scores['composite_score']}/100` (3s Hook `{p.scores['hook_3s']}`, Cliffhanger `{p.scores['cliffhanger_cut']}`, Cold Clarity `{p.scores['standalone_clarity']}`, Pacing `{p.scores['pacing_density']}`)")
        md_lines.append("- **Segments (in order):**")
        for s in p.segments:
            md_lines.append(
                f"  {s.segment_order}. **Episode {s.episode_number}** | Start: `{s.start_fmt}` | End: `{s.end_fmt}` (`{s.duration_sec:.2f}s`) — *[{s.segment_role}]* {s.key_action_or_line}"
            )
        md_lines.append(f"- **Opening 3-Second Hook:** {p.opening_hook_3s}")
        md_lines.append(f"- **Ending Cliffhanger Cut:** {p.ending_cliffhanger}")
        md_lines.append(f"- **Why It Works (1-Line):** {p.why_it_works_one_liner}")
        md_lines.append(f"- **Why Post Here in Funnel:** {p.posting_priority_reason}")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Part 2: Quality Check — Where the Tool Picks Badly & Guardrails Added",
        "",
    ])
    for fp in quality_audit["documented_failure_cases"]:
        md_lines.append(f"### `{fp['failure_id']}`: {fp['failure_archetype']} (Episodes {fp['episodes_involved']})")
        md_lines.append(f"- **Naive Window Picked:** `{fp['naive_timestamps']}` (`{fp['naive_duration_sec']}s`) — Naive Score: `{fp['naive_trigger_score']}`")
        md_lines.append(f"- **Why Naive Logic Picked It:** {fp['why_naive_model_picked_it']}")
        md_lines.append(f"- **Why It Fails on Video:** {fp['why_it_fails_as_a_real_promo']}")
        md_lines.append(f"- **Engineering Fix:** {fp['guardrail_added_to_fix_it']}")
        md_lines.append("")

    md_lines.extend([
        "---",
        "",
        "## Part 3: Full 50-Episode Continuous Promo Catalog (Best 20–60s Window Per Episode)",
        "",
        "| Episode | Window (`Start - End`) | Duration | Score | Opening 3s Hook | Ending Cliffhanger | Why It Works |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ])
    for ep_p in sorted(per_ep_promos, key=lambda x: x.segments[0].episode_number):
        s = ep_p.segments[0]
        hook_short = ep_p.opening_hook_3s.replace("|", "/").replace("\n", " ")[:90]
        cliff_short = ep_p.ending_cliffhanger.replace("|", "/").replace("\n", " ")[:90]
        why_short = ep_p.why_it_works_one_liner.replace("|", "/").replace("\n", " ")
        md_lines.append(
            f"| **Ep {s.episode_number:02d}** | `{s.start_fmt} - {s.end_fmt}` | `{ep_p.total_duration_sec:.1f}s` | `{ep_p.scores['composite_score']}` | {hook_short}... | {cliff_short}... | {why_short} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Part 4: Module 3A — Cross-Episode Speaker Collision & VLM ASR Entity Audit",
        "",
        f"- **Bride's Maiden House (Intentional):** `{entity_audit['canonical_show_bible']['bride_maiden_house']['canonical_name']}` (Present in Episodes {entity_audit['canonical_show_bible']['bride_maiden_house']['episodes_present']})",
        f"- **Groom's Dukedom (Canonical):** `{entity_audit['canonical_show_bible']['groom_dukedom_house']['canonical_name']}`",
        "- **8 Phonetic ASR Variants of 'Ashbourne' Found Across Screenplay Batches:**",
    ])
    for var_name, info in entity_audit["canonical_show_bible"]["groom_dukedom_house"]["asr_phonetic_mutations_detected"].items():
        md_lines.append(f"  - `{var_name}` -> Episodes `{info['episodes']}` ({info['note']})")

    (out_dir / "FULL_50_EPISODE_PROMO_REPORT.md").write_text("\n".join(md_lines), encoding="utf-8")


def run_full_pipeline(
    input_dir: Path = DEFAULT_INPUT_REL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    provider: str = "deterministic",
    model: str = "auto",
    api_key: str | None = None,
) -> dict:
    t0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)

    router = UniversalLLMRouter(LLMConfig(provider=provider, model=model, api_key=api_key))

    # 1. Parse all 50 episodes
    episodes = parse_all_episodes(input_dir)

    # 2. Run Module 3A: Entity & Speaker Alias Resolution (mutates canonical_speaker in-place)
    entity_audit = run_entity_and_continuity_audit(episodes)

    # 3. Run Module 3C: Pacing Heartbeat & Cross-Episode Recap Overlap Analysis
    pacing_audit = analyze_pacing_and_overlaps(episodes)

    # 4. Run Core Problem 1: Per-Episode Best Continuous Promos + Top 10 Ranked Flagship/Stitched Promos
    per_ep_promos = find_best_single_stretch_per_episode(episodes)
    top_promos = build_curated_flagship_and_stitched_promos(episodes, router=router)

    # Also rank all 50 per-episode promos by composite score
    per_ep_sorted = sorted(per_ep_promos, key=lambda x: x.scores["composite_score"], reverse=True)
    for idx, cand in enumerate(per_ep_sorted, start=1):
        cand.rank = idx

    # 5. Run Quality & Failure Auditor ("Where does your tool pick badly?")
    quality_audit = run_quality_and_failure_audit(episodes)

    # 6. Run Module 3B (IsoDub Timed Hindi/Indian Localization) & Module 3D (ShotForge Asset Pack) on Top 3 Promos
    localized_samples = [
        localize_promo_isochronous(p, episodes, router=router) for p in top_promos[:3]
    ]
    shot_packs = [
        compile_promo_shot_pack(p, episodes) for p in top_promos[:3]
    ]

    elapsed = round(time.perf_counter() - t0, 3)

    # Save unedited JSON outputs
    (output_dir / "01_top_ranked_promos.json").write_text(
        json.dumps([asdict(p) for p in top_promos], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "02_all_50_episodes_best_promos.json").write_text(
        json.dumps([asdict(p) for p in per_ep_promos], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "03_quality_check_and_bad_picks_audit.json").write_text(
        json.dumps(quality_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "04_module3A_entity_and_continuity_audit.json").write_text(
        json.dumps(entity_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "05_module3C_pacing_and_recap_overlaps.json").write_text(
        json.dumps(pacing_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "06_module3B_isodub_hindi_transcreation.json").write_text(
        json.dumps(localized_samples, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "07_module3D_shotforge_production_packs.json").write_text(
        json.dumps(shot_packs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save CSV catalog for easy spreadsheet opening
    with open(output_dir / "all_promos_ranked_catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Catalog_Group", "Rank", "Promo_ID", "Title", "Promo_Type", "Posting_Wave",
            "Total_Duration_Sec", "Segments_In_Order", "Opening_3s_Hook",
            "Ending_Cliffhanger", "Why_It_Works_1_Line", "Composite_Score",
        ])
        for p in top_promos:
            segs = " | ".join(f"Ep {s.episode_number} ({s.start_fmt}-{s.end_fmt})" for s in p.segments)
            writer.writerow([
                "FLAGSHIP_TOP_10", p.rank, p.promo_id, p.title, p.promo_type, p.posting_wave,
                p.total_duration_sec, segs, p.opening_hook_3s, p.ending_cliffhanger,
                p.why_it_works_one_liner, p.scores["composite_score"],
            ])
        for p in per_ep_sorted:
            segs = " | ".join(f"Ep {s.episode_number} ({s.start_fmt}-{s.end_fmt})" for s in p.segments)
            writer.writerow([
                "PER_EPISODE_50", p.rank, p.promo_id, p.title, p.promo_type, p.posting_wave,
                p.total_duration_sec, segs, p.opening_hook_3s, p.ending_cliffhanger,
                p.why_it_works_one_liner, p.scores["composite_score"],
            ])

    _write_markdown_report(
        output_dir, episodes, top_promos, per_ep_promos,
        quality_audit, entity_audit, pacing_audit, elapsed, router
    )

    return {
        "episodes_count": len(episodes),
        "total_blocks": sum(len(e.blocks) for e in episodes),
        "elapsed_sec": elapsed,
        "top_promos_count": len(top_promos),
        "per_ep_promos_count": len(per_ep_promos),
        "anomalies_count": entity_audit["total_anomalies_found"],
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kuku PromoStudio pipeline on microdrama screenplays.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT_REL), help="Folder containing episode .txt files")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory for raw_outputs")
    parser.add_argument("--provider", type=str, default="deterministic", choices=["deterministic", "gemini", "groq", "openrouter", "mistral", "openai"])
    parser.add_argument("--model", type=str, default="auto", help="Model name override")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key for live LLM pass")
    args = parser.parse_args()

    summary = run_full_pipeline(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
    )
    print(json.dumps(summary, indent=2))
