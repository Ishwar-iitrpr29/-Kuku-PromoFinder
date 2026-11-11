# 🎬 Kuku PromoStudio — AI Microdrama Promo Finder & Script Intelligence Suite

Submission for **Kuku AI Generalist Assignment — Problem 1: Find the Promos (Ads)**, enhanced with integrated pre-production & localization modules (**3A: Cross-Episode Speaker & VLM Entity Resolver**, **3B: IsoDub Timestamp-Locked Hindi/Indian Promo Adapter**, **3C: CliffPulse Retention Heartbeat & Recap-Overlap Auditor**, and **3D: ShotForge Production Asset Compiler**).

Tested and verified across all **50 screenplay episodes (3,727 timestamped beats / ~85 minutes of footage)** of *"I Married a Broken Soldier. He was a Duke"*.

---

## 🚀 Quick Start (Runs Out-of-the-Box in < 2 Seconds)

### 1. Launch Interactive Studio UI (Streamlit)
```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```
*(Note: Using `python -m streamlit run app.py` instead of `streamlit run app.py` avoids Windows Application Control / Smart App Control policies blocking unsigned pip `.exe` wrapper shims.)*

- **Backend `.env` & Streamlit `st.secrets` Authentication (Zero Keys Exposed in UI):** Add your `GEMINI_API_KEY` (or `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`) to `.env` locally or to Streamlit Cloud `Secrets`. API keys are never shown or editable in the UI.
- **Dynamic Router, Model & Strategy Switching (Auto Re-Runs on Change):** Open `⚙️ AI Model Settings` in the left sidebar to select from active `.env` routers (`gemini`, `groq`, `openrouter`, `mistral`, `openai`), choose specific models (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.0-flash`, `llama-3.3-70b-versatile`, etc.), and switch creative profiles (`Standard Hybrid`, `Bollywood Dramatic Microdrama`, `Conversational Urban Hinglish`)—automatically re-running and updating scores, Hindi/Hinglish dubs, and video prompts while keeping all deterministic guardrails active.
- **Custom Screenplay Paste & Input Validity Gate:** Switch the sidebar source to `"✍️ Paste / Test Custom Script"` to test unformatted plain-text scripts (`Speaker: Dialogue` or `[ACTION]`—auto-timed at `~2.4 words/sec`), `<20s` short scenes (`SCRIPT_TOO_SHORT_FOR_PROMO`), off-topic non-screenplay text (`NOT_A_SCREENPLAY`), or random gibberish (`GIBBERISH_INPUT`).

### 2. Run CLI Batch Pipeline on Any Screenplay Folder
```powershell
# Run on the default 50-episode Kuku input pack (< 2.0s execution)
python run_pipeline.py

# Run on a brand-new screenplay folder during your interview
python run_pipeline.py --input "D:\path\to\new_show_scripts" --output "raw_outputs_new"

# Run with a live Free-Tier LLM API (e.g., Gemini or Groq)
python run_pipeline.py --provider gemini --api-key YOUR_GEMINI_KEY
python run_pipeline.py --provider groq --api-key YOUR_GROQ_KEY
```

### 3. Run Automated UI Tests
```powershell
python -m pytest test_app_ui.py
```

---

## 🖥️ Clean 3-Tab Studio Layout

The Streamlit UI has been streamlined into a clean, distraction-free **3-tab layout**:

1. **Tab 1: 🏆 Top 10 Promos & Stitched Trailers**
   - Lists the Top 10 flagship acquisition promos (Single-Stretch and Multi-Episode Stitched).
   - Expanding any promo reveals 4 focused sub-tabs:
     - 🎬 **Beat-by-Beat Script:** Full timestamped script breakdown with camera angles, sound effects, action directions, and resolved speaker names.
     - 🎙️ **Hindi Dub (IsoDub):** Syllable-budgeted Hindi dubbing script strictly capped at $\le 4.6\text{ syllables/sec}$ so voiceover never overflows camera cuts.
     - 📋 **Shot List & Visuals:** Shot-by-shot Midjourney/FLUX prompts, episode-persistent wardrobe tags (e.g. bandaged hands in Ep 14–17, scullery maid disguise in Ep 22–28), and ElevenLabs voice IDs.
     - 🔬 **Evaluation & Pacing:** Tension progression chart, hook velocity score, cut cliffhanger rating, and trope clarity score.
2. **Tab 2: 📊 All 50 Episodes Explorer**
   - Select any episode (Ep 1 to Ep 50).
   - Visualizes the full-episode tension curve and automatically isolates the highest-scoring continuous 20–60s ad cut.
3. **Tab 3: 🛡️ QA & False-Positive Traps**
   - Contrastive failure analysis documenting the **5 real-world traps** tested and fixed:
     - `FAIL-01` (Cross-Episode Recap Trap): Jaccard overlap auto-trims repeated altar rejection between Ep 1 and Ep 2.
     - `FAIL-02` (Dark Forest Trap): Ep 25 mentions poison but shows 39s of static walking—penalized by visual kinetic filter.
     - `FAIL-03` (Dialogue without Visual Hook): Rejects talking heads without a physical/kinetic action trigger in the opening 3 seconds.
     - `FAIL-04` (Pronoun Ambiguity): Penalizes clips opening with ambiguous pronouns ("Did you hear what she did?") that disorient cold viewers.
     - `FAIL-05` (Entity Drift & Collision): Resolves 53 lines where VLM tagged both Lord Ashcroft and villain cousin Rupert as `BLOND NOBLEMAN`, and fixes 8 phonetic misspellings of Duke Ashbourne.
   - Includes the complete Entity Linter table.

---

## 📂 Repository & Submission Contents

| Path | Description |
| :--- | :--- |
| [`Solution_Doc_Problem1.pdf`](./Solution_Doc_Problem1.pdf) | **Required 2-Page PDF Solution Document** (with blank candidate header) covering problem framing, assumptions, hybrid code+AI architecture, scale/cost math, quality checking, and failure analysis. |
| [`Demo_Walkthrough_Problem1_PromoFinder.mp4`](./Demo_Walkthrough_Problem1_PromoFinder.mp4) | **Full HD (`1920x1080`) Walkthrough Video (< 3 min)** with Neural Voiceover, burned-in broadcast captions, and companion [`.srt` subtitles](./Demo_Walkthrough_Problem1_PromoFinder.srt). |
| [`app.py`](./app.py) | Full Streamlit studio interface with 3 clean tabs, sidebar screenplay loader, and model switcher. |
| [`run_pipeline.py`](./run_pipeline.py) | High-speed CLI pipeline that parses all 50 episodes (3,727 beats) in `<2.0s`. |
| [`generate_solution_doc.py`](./generate_solution_doc.py) | ReportLab script that compiles the exact 2-page PDF solution brief. |
| [`test_app_ui.py`](./test_app_ui.py) | Streamlit `AppTest` regression test suite. |
| [`raw_outputs/`](./raw_outputs/) | **Required Unedited Outputs Folder** generated across all 50 episodes. |
| `raw_outputs/FULL_50_EPISODE_PROMO_REPORT.md` | Human-readable report of Top 10 Ranked Promos, 50-Episode Continuous Catalog, and Failure Audit. |
| `raw_outputs/all_promos_ranked_catalog.csv` | Spreadsheet-ready CSV containing all ranked promos with exact start/end timestamps. |
| `raw_outputs/01_top10_flagship_promos.json` | Unedited JSON of Top 10 Flagship Single-Stretch & Cross-Episode Stitched Promos. |
| `raw_outputs/02_all_50_episodes_promos.json` | Unedited JSON of the best 20–60s continuous promo inside every episode. |
| `raw_outputs/03_module3A_entity_and_speaker_audit.json` | Full audit of 260 VLM transcription anomalies (8 phonetic spellings + `BLOND NOBLEMAN` collision). |
| `raw_outputs/04_module3B_3D_localized_production_packs.json` | Syllable-locked Hindi dubbing and ShotForge AI video/TTS prompts. |
| `raw_outputs/05_module3C_pacing_and_recap_audit.json` | 5-second rolling tension curves, dead-air zones, and cross-episode recap overlap detection. |
| `raw_outputs/06_quality_check_failure_modes.json` | Complete audit of the 5 bad-pick archetypes blocked by our guardrails. |
| `raw_outputs/07_pipeline_summary_metrics.json` | End-to-end execution telemetry and corpus-level statistics. |

---

## ⚙️ Architecture & Code Map (`src/`)

- [`src/parser.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/parser.py): Robust line-by-line regex parser extracting timestamped `ScriptBlock` objects with hook, cliffhanger, conflict, and kinetic visual scores.
- [`src/promo_finder.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/promo_finder.py): Sliding-window composite evaluator ($0.30 \times \text{Hook}_{3\text{s}} + 0.30 \times \text{Cliffhanger}_{\text{end}} + 0.25 \times \text{Clarity} + 0.15 \times \text{Density}$) and multi-episode stitched trailer assembler.
- [`src/entity_resolver.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/entity_resolver.py): Module 3A VLM disambiguator; maps appearance tags, fixes Ashbourne phonetic drift, and resolves Lord Ashcroft vs Rupert.
- [`src/pacing_analyzer.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/pacing_analyzer.py): Module 3C retention heartbeat analyzer; calculates rolling 5s impulse tension, flags $>20\text{s}$ dead-air, and computes Jaccard recap overlap.
- [`src/localizer.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/localizer.py): Module 3B IsoDub adapter; calculates duration-locked syllable budgets ($\le 4.6\text{ syllables/sec}$) and applies curated dramatic Hindi translations.
- [`src/shot_compiler.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/shot_compiler.py): Module 3D asset compiler; generates Midjourney/FLUX camera prompts, persistent character wardrobe states, and ElevenLabs voice presets.
- [`src/llm_refiner.py`](file:///d:/projects/KukuFM/project1_promo_engine/src/llm_refiner.py): Swappable LLM integration layer supporting Gemini, Groq, OpenRouter, Mistral, and OpenAI.
