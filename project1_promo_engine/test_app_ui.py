import os
import sys
from streamlit.testing.v1 import AppTest


def test_project1_streamlit_ui():
    print("Testing Project 1 Streamlit App UI...")
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    if proj_dir not in sys.path:
        sys.path.insert(0, proj_dir)
    app_path = os.path.join(proj_dir, "app.py")
    at = AppTest.from_file(app_path, default_timeout=25)
    at.run()
    assert not at.exception, f"App failed to run: {at.exception}"
    print(">> App loaded successfully.")

    # Verify metrics
    print(f">> Metric count: {len(at.metric)}")
    metrics = {m.label: m.value for m in at.metric}
    print("   Metrics:", metrics)
    assert metrics["Episodes Analyzed"] == "50"
    assert metrics["Top Flagship Promos"] == "10"
    assert metrics["Per-Episode Cuts"] == "50"

    # Find the ranked promo selectbox (in the main area, its label is 'Choose a ranked promo to view details:')
    promo_sb = next(sb for sb in at.selectbox if "Choose a ranked promo" in sb.label)
    print(f">> Current selected promo: {promo_sb.value}")
    options = promo_sb.options
    assert len(options) >= 10
    promo_sb.select(options[1])
    at.run()
    assert not at.exception, f"App crashed after picking Rank #2: {at.exception}"
    print(f">> Selected Rank #2 successfully: {promo_sb.value}")

    # Test switching AI Router, Select Model, and Model Engine Profile
    router_sb = next(sb for sb in at.selectbox if "AI Router" in sb.label)
    assert all("deterministic" not in opt.lower() for opt in router_sb.options), "Deterministic mode should not be shown in UI"
    llm_model_sb = next(sb for sb in at.selectbox if "Select Model" in sb.label)
    assert "gemini-2.5-flash" in llm_model_sb.options and "gemini-2.5-pro" in llm_model_sb.options

    model_sb = next(sb for sb in at.selectbox if "Model Engine Profile" in sb.label)
    assert all("deterministic" not in opt.lower() for opt in model_sb.options), "Deterministic mode should not be shown in UI"
    assert not any("api key" in ti.label.lower() for ti in at.text_input), "API Key input should not be shown in UI"
    default_success = [s.value for s in at.success]
    model_sb.select("Bollywood Dramatic Microdrama")
    at.run()
    assert not at.exception, f"App crashed after switching to bollywood-dramatic: {at.exception}"
    dramatic_success = [s.value for s in at.success]
    assert any("High-Intensity Bollywood Microdrama" in s for s in dramatic_success), "Expected dramatic profile tag in Why This Cut Converts"
    assert default_success != dramatic_success, "Expected model switch to dynamically update promo output!"

    model_sb = next(sb for sb in at.selectbox if "Model Engine Profile" in sb.label)
    model_sb.select("Conversational Urban Hinglish")
    at.run()
    assert not at.exception
    hinglish_success = [s.value for s in at.success]
    assert any("Modern Urban Hinglish Reels" in s for s in hinglish_success)
    print(">> Verified dynamic Model Engine Profile switching across all 3 profiles!")

    # Test Custom Script Mode (Plain-Text Scene, Short Scene, Off-Topic Biryani, Gibberish)
    source_radio = at.radio[0]
    source_radio.set_value("✍️ Paste / Test Custom Script")
    at.run()
    assert not at.exception, f"App crashed entering Custom Script mode: {at.exception}"
    metrics_custom = {m.label: m.value for m in at.metric}
    assert metrics_custom["Episodes Analyzed"] == "1"
    assert metrics_custom["Top Flagship Promos"] == "1"
    print(">> Custom Unformatted Plain-Text Scene parsed, auto-timed, and scored!")

    # Switch preset to CT02: Too-Short Scene (<20s)
    preset_sb = next(sb for sb in at.selectbox if "Quick Test Scenario" in sb.label)
    preset_sb.select("CT02: Too-Short Scene (<20s Minimum Ad Rule)")
    at.run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("Scene Under 20-Second Standalone Ad Minimum" in w for w in warnings), "Expected <20s warning banner!"
    print(">> Too-Short Scene (<20s) warning banner verified!")

    # Switch preset to CT03: Off-Topic Non-Screenplay (Cooking Recipe)
    preset_sb = next(sb for sb in at.selectbox if "Quick Test Scenario" in sb.label)
    preset_sb.select("CT03: Off-Topic Non-Screenplay Input (Cooking Recipe)")
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("NOT_A_SCREENPLAY" in e for e in errors), "Expected NOT_A_SCREENPLAY error banner!"
    print(">> Off-Topic Non-Screenplay input cleanly rejected!")

    # Switch preset to CT04: Random Gibberish
    preset_sb = next(sb for sb in at.selectbox if "Quick Test Scenario" in sb.label)
    preset_sb.select("CT04: Random Gibberish / Keysmash")
    at.run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("GIBBERISH_INPUT" in e for e in errors), "Expected GIBBERISH_INPUT error banner!"
    print(">> Random Gibberish input cleanly rejected!")

    print("ALL PROJECT 1 STREAMLIT TESTS PASSED!")


if __name__ == "__main__":
    test_project1_streamlit_ui()
