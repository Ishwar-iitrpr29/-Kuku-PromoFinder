"""
Deterministic Screenplay Parser & Beat Scorer for Kuku Microdramas.
Parses 50+ timestamped episode scripts in <0.5 seconds with zero API cost.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


TIMESTAMP_LINE_RE = re.compile(
    r"Start:\s*\((\d{1,2}):(\d{2}(?:\.\d+)?)\)\s*\|\s*End:\s*\((\d{1,2}):(\d{2}(?:\.\d+)?)\)",
    re.IGNORECASE,
)
BLOCK_HEADER_RE = re.compile(r"^\*\*([^*]+)\*\*:?\s*$")
SCENE_NUM_RE = re.compile(r"^SCENE\s+(\d+)", re.IGNORECASE)
SCENE_HEADING_RE = re.compile(r"^Scene heading:\s*(.+)$", re.IGNORECASE)
SCENE_START_RE = re.compile(r"^Scene start:\s*(\d{1,2}):(\d{2}(?:\.\d+)?)", re.IGNORECASE)
SCENE_END_RE = re.compile(r"^Scene end:\s*(\d{1,2}):(\d{2}(?:\.\d+)?)", re.IGNORECASE)
BATCH_ID_RE = re.compile(r"Episode\s+(\d+)", re.IGNORECASE)

# High-velocity visual & dramatic hook triggers
VISUAL_SHOCK_KEYWORDS = {
    "slap": 3.2,
    "strike": 2.6,
    "strikes": 2.6,
    "struck": 2.4,
    "riding crop": 3.0,
    "riding-crop": 3.0,
    "whip": 2.8,
    "shackles": 3.0,
    "chained": 2.8,
    "chains": 2.6,
    "handcuffs": 2.8,
    "under arrest": 3.0,
    "arrest": 2.5,
    "explosion": 3.5,
    "detonating": 3.2,
    "shattered": 2.5,
    "blaze": 3.0,
    "ablaze": 3.0,
    "fire": 2.4,
    "flames": 2.6,
    "inferno": 3.0,
    "red-hot": 3.2,
    "branding iron": 3.4,
    "iron": 2.0,
    "blood": 2.6,
    "bloody": 2.6,
    "wound": 2.2,
    "laceration": 2.6,
    "blackthorn": 2.6,
    "thorns": 2.4,
    "kneels": 2.5,
    "one knee": 3.0,
    "marry me": 3.2,
    "marrying me": 3.2,
    "altar": 2.2,
    "bridal gown": 2.2,
    "wedding": 2.0,
    "Duke": 2.4,
    "Duchess": 2.4,
    "treason": 2.6,
    "witchcraft": 2.6,
    "gallows": 2.8,
    "execution": 2.8,
    "hang": 2.6,
    "dagger": 2.8,
    " blade": 2.8,
    "sword": 2.6,
    "pistol": 2.8,
    "mud": 2.2,
    "throws": 2.2,
    "shoves": 2.4,
    "lunges": 2.5,
    "choking": 2.4,
    "gasps": 1.8,
}

HOOK_DIALOGUE_PATTERNS = [
    (re.compile(r"you should turn around", re.I), 2.8),
    (re.compile(r"marrying me|cannot marry you|changed my mind", re.I), 3.2),
    (re.compile(r"under arrest|seize (her|him)|take (her|him) to the", re.I), 3.0),
    (re.compile(r"who did this|planted it|stole my", re.I), 2.6),
    (re.compile(r"rightful duke|duchess of|my lawful wife", re.I), 3.1),
    (re.compile(r"wait\.|stop!|enough!", re.I), 2.4),
    (re.compile(r"burn|hang|gallows|witch|poison", re.I), 2.7),
    (re.compile(r"crawl|bare hands|on your knees|scullery maid", re.I), 2.9),
]

CLIFFHANGER_CUES = {
    "cliffhanger": 3.5,
    "cut to black": 3.0,
    "ominous": 2.6,
    "swells": 2.0,
    "suddenly": 2.2,
    "bursts open": 3.0,
    "swings open": 2.6,
    "latch turns": 2.6,
    "freezes": 2.2,
    "stunned": 2.2,
    "speechless": 2.2,
    "shock": 2.2,
    "gasp": 2.0,
    "defiance": 2.0,
    "explosion": 3.5,
    "invalid": 2.8,
    "caught": 2.5,
}

RESOLUTION_PENALTY_CUES = {
    "resolving chord": -2.5,
    "peaceful": -2.0,
    "warmth": -1.5,
    "gentle smile": -1.5,
    "falls asleep": -2.0,
    "quiet walk": -1.8,
}

HIGH_INTENSITY_EMOTIONS = {
    "cold", "callous", "cruel", "gloating", "smug", "outraged", "accusing",
    "betrayed", "furious", "seething", "dangerous", "venomous", "mocking",
    "spiteful", "imperious", "commanding", "terrified", "horrified", "defiant",
    "scathing", "desperate", "pleading", "shouting", "enraged", "vindictive",
    "sadistic", "tormented", "treacherous", "chilling",
}


def parse_mmss(minutes_str: str, seconds_str: str) -> float:
    return int(minutes_str) * 60.0 + float(seconds_str)


def format_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    mins = int(seconds // 60)
    secs = seconds - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"


@dataclass
class ScriptBlock:
    block_index: int
    scene_number: int
    block_type: str  # CAMERA | ACTION | SOUND FX | DIALOGUE
    raw_speaker: str
    canonical_speaker: str
    emotion_cues: List[str]
    start_sec: float
    end_sec: float
    text: str
    line_number: int
    hook_score: float = 0.0
    cliffhanger_score: float = 0.0
    conflict_score: float = 0.0
    visual_kinetic_score: float = 0.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, round(self.end_sec - self.start_sec, 2))


@dataclass
class SceneInfo:
    scene_number: int
    heading: str
    start_sec: float
    end_sec: float


@dataclass
class EpisodeScript:
    episode_number: int
    filename: str
    batch_id: str
    logline: str
    continuity_notes: str
    total_duration_sec: float
    scenes: List[SceneInfo] = field(default_factory=list)
    blocks: List[ScriptBlock] = field(default_factory=list)
    input_status: str = "VALID_TIMESTAMPED_SCRIPT"
    status_message: str = "Timestamped screenplay parsed cleanly."

    def to_summary_dict(self) -> Dict:
        return {
            "episode_number": self.episode_number,
            "filename": self.filename,
            "batch_id": self.batch_id,
            "logline": self.logline,
            "continuity_notes": self.continuity_notes,
            "total_duration_sec": round(self.total_duration_sec, 2),
            "total_duration_fmt": format_mmss(self.total_duration_sec),
            "scene_count": len(self.scenes),
            "block_count": len(self.blocks),
            "input_status": self.input_status,
            "status_message": self.status_message,
        }


def _score_block(block: ScriptBlock) -> None:
    text_lower = block.text.lower()
    hook = 0.0
    cliff = 0.0
    conflict = 0.0
    visual = 0.0

    # Keyword scoring
    for kw, weight in VISUAL_SHOCK_KEYWORDS.items():
        if kw.lower() in text_lower:
            hook += weight * 0.7
            conflict += weight * 0.5
            if block.block_type in ("ACTION", "CAMERA", "SOUND FX"):
                visual += weight * 0.8

    if block.block_type == "DIALOGUE":
        for pat, weight in HOOK_DIALOGUE_PATTERNS:
            if pat.search(block.text):
                hook += weight
                conflict += weight * 0.7
        if "?" in block.text or "!" in block.text:
            cliff += 1.2
            conflict += 0.8
        for emo in block.emotion_cues:
            for token in re.split(r"[\s,/]+", emo.lower()):
                if token in HIGH_INTENSITY_EMOTIONS:
                    conflict += 1.4
                    hook += 0.8

    elif block.block_type == "CAMERA":
        if "extreme close-up" in text_lower or "tight close-up" in text_lower:
            visual += 1.8
            cliff += 1.2
        elif "close-up" in text_lower:
            visual += 1.1
        if "low-angle" in text_lower or "push-in" in text_lower:
            visual += 1.2
            hook += 0.8
        if "cut to black" in text_lower:
            cliff += 3.0

    elif block.block_type == "ACTION":
        visual += 1.5
        if any(w in text_lower for w in ("walks slowly", "walking slowly", "quiet walk", "sits quietly")):
            visual -= 1.2

    elif block.block_type == "SOUND FX":
        visual += 1.4
        cliff += 1.8

    for cue, weight in CLIFFHANGER_CUES.items():
        if cue in text_lower:
            cliff += weight

    for cue, penalty in RESOLUTION_PENALTY_CUES.items():
        if cue in text_lower:
            cliff += penalty

    block.hook_score = round(max(0.0, hook), 2)
    block.cliffhanger_score = round(max(0.0, cliff), 2)
    block.conflict_score = round(max(0.0, conflict), 2)
    block.visual_kinetic_score = round(max(0.0, visual), 2)


def parse_episode_file(filepath: Path) -> EpisodeScript:
    lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    ep_num = 0
    ep_match = re.search(r"episode-(\d+)", filepath.name, re.I)
    if ep_match:
        ep_num = int(ep_match.group(1))

    batch_match = BATCH_ID_RE.search(filepath.name)
    batch_id = batch_match.group(1) if batch_match else "unknown"

    logline = ""
    continuity = ""
    scenes: List[SceneInfo] = []
    blocks: List[ScriptBlock] = []

    current_scene_num = 1
    current_scene_heading = ""
    current_scene_start = 0.0
    current_scene_end = 0.0

    i = 0
    n = len(lines)
    block_idx = 0

    while i < n:
        line = lines[i].strip()
        if line.upper().startswith("EPISODE NUMBER:") and ep_num == 0:
            m = re.search(r"(\d+)", line)
            if m:
                ep_num = int(m.group(1))
        elif line.upper().startswith("EPISODE LOGLINE:"):
            logline = line.split(":", 1)[1].strip()
        elif line.upper().startswith("CONTINUITY NOTES:"):
            continuity = line.split(":", 1)[1].strip()
        elif SCENE_NUM_RE.match(line):
            m = SCENE_NUM_RE.match(line)
            current_scene_num = int(m.group(1))
            current_scene_heading = ""
            current_scene_start = 0.0
            current_scene_end = 0.0
        elif SCENE_HEADING_RE.match(line):
            current_scene_heading = SCENE_HEADING_RE.match(line).group(1).strip()
        elif SCENE_START_RE.match(line):
            m = SCENE_START_RE.match(line)
            current_scene_start = parse_mmss(m.group(1), m.group(2))
        elif SCENE_END_RE.match(line):
            m = SCENE_END_RE.match(line)
            current_scene_end = parse_mmss(m.group(1), m.group(2))
            scenes.append(
                SceneInfo(
                    scene_number=current_scene_num,
                    heading=current_scene_heading,
                    start_sec=current_scene_start,
                    end_sec=current_scene_end,
                )
            )
        else:
            header_match = BLOCK_HEADER_RE.match(line)
            if header_match:
                raw_header = header_match.group(1).strip().rstrip(":").strip()
                header_line_num = i + 1
                i += 1
                while i < n and not lines[i].strip():
                    i += 1
                start_sec, end_sec = 0.0, 0.0
                if i < n:
                    ts_match = TIMESTAMP_LINE_RE.search(lines[i])
                    if ts_match:
                        start_sec = parse_mmss(ts_match.group(1), ts_match.group(2))
                        end_sec = parse_mmss(ts_match.group(3), ts_match.group(4))
                        i += 1
                text_lines = []
                while i < n:
                    peek = lines[i].strip()
                    if not peek:
                        i += 1
                        break
                    if BLOCK_HEADER_RE.match(peek) or SCENE_NUM_RE.match(peek) or peek.startswith("--- SCRIPT END"):
                        break
                    text_lines.append(peek)
                    i += 1

                content_text = " ".join(text_lines).strip()
                upper_h = raw_header.upper()
                if upper_h == "CAMERA":
                    b_type = "CAMERA"
                    speaker = ""
                    emotions: List[str] = []
                elif upper_h == "ACTION":
                    b_type = "ACTION"
                    speaker = ""
                    emotions = []
                elif upper_h in ("SOUND FX", "SFX"):
                    b_type = "SOUND FX"
                    speaker = ""
                    emotions = []
                else:
                    b_type = "DIALOGUE"
                    paren_parts = re.findall(r"\(([^)]+)\)", raw_header)
                    speaker = re.sub(r"\([^)]+\)", "", raw_header).strip().rstrip(":").strip()
                    emotions = [
                        p.strip()
                        for p in paren_parts
                        if p.strip().upper() not in ("O.S.", "OS", "OFF-SCREEN", "V.O.", "VO", "CONTINUES", "O.S. / CONTINUES")
                    ]

                blk = ScriptBlock(
                    block_index=block_idx,
                    scene_number=current_scene_num,
                    block_type=b_type,
                    raw_speaker=speaker,
                    canonical_speaker=speaker,
                    emotion_cues=emotions,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    text=content_text,
                    line_number=header_line_num,
                )
                _score_block(blk)
                blocks.append(blk)
                block_idx += 1
                continue
        i += 1

    if ep_num == 0:
        ep_num = 1

    input_status = "VALID_TIMESTAMPED_SCRIPT"
    status_message = "Timestamped screenplay parsed cleanly."

    # If standard **HEADER** blocks exist but lacked Start/End timestamps, synthesize sequential timestamps
    if blocks and all(b.end_sec <= 0.01 for b in blocks):
        cursor = 0.0
        for b in blocks:
            wc = max(1, len(b.text.split()))
            dur_est = max(2.2, round(wc / 2.5, 1))
            b.start_sec = round(cursor, 2)
            b.end_sec = round(cursor + dur_est, 2)
            cursor = b.end_sec
        input_status = "SYNTHESIZED_PLAIN_SCRIPT"
        status_message = "Screenplay lacked explicit (MM:SS) timestamps; sequential speech/action timing was synthesized automatically."

    # Fallback parser & validity gate when no **HEADER** markdown blocks were found
    if not blocks:
        raw_full = "\n".join(lines).strip()
        fallback_blocks, input_status, status_message = _fallback_parse_or_reject(raw_full)
        blocks = fallback_blocks

    total_dur = 0.0
    if scenes:
        total_dur = max((s.end_sec for s in scenes), default=0.0)
    if blocks:
        total_dur = max(total_dur, max((b.end_sec for b in blocks), default=0.0))

    if blocks and total_dur < 20.0 and input_status in ("VALID_TIMESTAMPED_SCRIPT", "SYNTHESIZED_PLAIN_SCRIPT"):
        input_status = "SCRIPT_TOO_SHORT_FOR_PROMO"
        status_message = (
            f"Valid screenplay scene detected ({total_dur:.1f}s across {len(blocks)} beats), "
            "but total duration is under the 20.0-second minimum required for a standalone microdrama ad cut."
        )

    return EpisodeScript(
        episode_number=ep_num,
        filename=filepath.name,
        batch_id=batch_id,
        logline=logline,
        continuity_notes=continuity,
        total_duration_sec=total_dur,
        scenes=scenes,
        blocks=blocks,
        input_status=input_status,
        status_message=status_message,
    )


PLAIN_DIALOGUE_RE = re.compile(
    r"^([A-Z][A-Za-z0-9\s\.'\-]{1,38}?)(?:\s*\(([^)]+)\))?\s*:\s*(.+)$"
)
OFF_TOPIC_KEYWORDS_RE = re.compile(
    r"\b(biryani|recipe|chicken|masala|tablespoon|preheat oven|boil water|def\s+\w+\(|import\s+numpy|console\.log|select\s+\*\s+from|weather forecast|cricket score|ipl match|stock price|bitcoin)\b",
    re.I,
)


def _is_gibberish_text(text: str) -> bool:
    cleaned = text.strip()
    if not cleaned or len(cleaned) < 10:
        return True
    alpha_chars = [c.lower() for c in cleaned if c.isalpha()]
    if len(alpha_chars) < 6:
        return True
    vowels = sum(1 for c in alpha_chars if c in "aeiou")
    vowel_ratio = vowels / max(1, len(alpha_chars))
    if vowel_ratio < 0.16 or vowel_ratio > 0.82:
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", cleaned.lower()):
        return True
    if re.search(r"(asdf|qwerty|zxcv|hjkl|12345|99999)", cleaned.lower()):
        return True
    words = re.findall(r"[A-Za-z]+", cleaned)
    if len(words) < 4:
        return True
    return False


def _fallback_parse_or_reject(raw_text: str) -> Tuple[List[ScriptBlock], str, str]:
    """
    Attempts to parse plain-text screenplay lines (e.g. 'Juliana (furious): Why did you betray me?'
    or '[ACTION] Doors burst open'). Rejects gibberish or off-topic non-screenplay prose.
    """
    if _is_gibberish_text(raw_text):
        return (
            [],
            "GIBBERISH_INPUT",
            "Input contains random gibberish, keysmashes, or too few words to form a screenplay.",
        )

    if OFF_TOPIC_KEYWORDS_RE.search(raw_text):
        return (
            [],
            "NOT_A_SCREENPLAY",
            "Input appears to be off-topic non-screenplay text (e.g., cooking recipe, code, or general query) with no dramatic dialogue or camera beats.",
        )

    parsed_blocks: List[ScriptBlock] = []
    cursor = 0.0
    b_idx = 0

    for line_no, raw_line in enumerate(raw_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.upper().startswith(("EPISODE", "SCENE", "LOGLINE", "CONTINUITY")):
            continue

        # Action or Camera bracketed line: [ACTION] ... or [CAMERA] ... or (Action ...)
        if line.startswith("[") and "]" in line:
            tag, rest = line[1:].split("]", 1)
            tag_up = tag.strip().upper()
            b_type = "CAMERA" if "CAM" in tag_up or "SHOT" in tag_up else ("SOUND FX" if "SFX" in tag_up or "SOUND" in tag_up else "ACTION")
            content = rest.strip()
            if content:
                wc = max(1, len(content.split()))
                dur = max(2.2, round(wc / 2.6, 1))
                blk = ScriptBlock(
                    block_index=b_idx,
                    scene_number=1,
                    block_type=b_type,
                    raw_speaker="",
                    canonical_speaker="",
                    emotion_cues=[],
                    start_sec=round(cursor, 2),
                    end_sec=round(cursor + dur, 2),
                    text=content,
                    line_number=line_no,
                )
                _score_block(blk)
                parsed_blocks.append(blk)
                cursor += dur
                b_idx += 1
            continue

        m = PLAIN_DIALOGUE_RE.match(line)
        if m:
            raw_hdr = m.group(1).strip()
            emo_raw = m.group(2) or ""
            content = m.group(3).strip()
            hdr_up = raw_hdr.upper()
            if hdr_up in ("CAMERA", "SHOT"):
                b_type = "CAMERA"
                speaker = ""
                emotions = []
            elif hdr_up in ("ACTION", "STAGE", "DIRECTION"):
                b_type = "ACTION"
                speaker = ""
                emotions = []
            elif hdr_up in ("SFX", "SOUND", "SOUND FX", "AUDIO"):
                b_type = "SOUND FX"
                speaker = ""
                emotions = []
            else:
                b_type = "DIALOGUE"
                speaker = raw_hdr.upper()
                emotions = [e.strip() for e in emo_raw.split(",") if e.strip()] if emo_raw else []

            wc = max(1, len(content.split()))
            dur = max(2.5, round(wc / 2.4, 1))
            blk = ScriptBlock(
                block_index=b_idx,
                scene_number=1,
                block_type=b_type,
                raw_speaker=speaker,
                canonical_speaker=speaker,
                emotion_cues=emotions,
                start_sec=round(cursor, 2),
                end_sec=round(cursor + dur, 2),
                text=content,
                line_number=line_no,
            )
            _score_block(blk)
            parsed_blocks.append(blk)
            cursor += dur
            b_idx += 1

    # Require at least 2 structured screenplay blocks (with at least 1 dialogue or action beat)
    if len(parsed_blocks) < 2:
        return (
            [],
            "NOT_A_SCREENPLAY",
            "No valid screenplay format (`**SPEAKER**:` or `Speaker: Dialogue` / `ACTION:`) found in input. Please provide a dramatic script with speaker tags or camera/action lines.",
        )

    return (
        parsed_blocks,
        "SYNTHESIZED_PLAIN_SCRIPT",
        f"Parsed {len(parsed_blocks)} plain-text screenplay beats and synthesized {cursor:.1f}s of sequential timing.",
    )


def parse_raw_script_text(raw_text: str, filename: str = "custom_episode_1.txt") -> EpisodeScript:
    """Helper to parse pasted screenplay text directly via a temporary file."""
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="kuku_raw_"))
    tmp_file = tmp_dir / filename
    tmp_file.write_text(raw_text, encoding="utf-8")
    return parse_episode_file(tmp_file)


def parse_all_episodes(folder_path: str | Path) -> List[EpisodeScript]:
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Screenplay directory not found: {folder}")

    txt_files = sorted(folder.glob("*.txt"))
    episodes = [parse_episode_file(f) for f in txt_files]
    episodes.sort(key=lambda e: (e.episode_number, e.filename))
    return episodes
