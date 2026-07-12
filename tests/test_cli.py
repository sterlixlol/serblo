"""Tests for serblo.generate.run_cli: the money path.

Covers the two subtle, spend-critical pieces: cline's --json cost/text parsing (is totalCost
cumulative-last-wins or per-turn-sum?) and the PERSISTENT cost ledger that makes --max-cost a
lifetime cap instead of a per-run one. Runs under pytest or standalone.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from serblo.generate import run_cli

# Real cline --json shape: reasoning -> denied tool (serbian-voice) -> text.
# TWO usage events whose totalCost is CUMULATIVE (0.047 then 0.052 = 0.047 + 0.005), so keeping the
# LAST totalCost is correct; a running += would double-count to ~0.099. This fixture pins that.
CLINE_SAMPLE = "\n".join([
    '{"ts":"t","type":"hook_event","hookEventName":"agent_start"}',
    '{"type":"agent_event","event":{"type":"content_start","contentType":"reasoning","reasoning":"I"}}',
    '{"type":"agent_event","event":{"type":"content_end","contentType":"reasoning","reasoning":"I need the skill."}}',
    '{"type":"agent_event","event":{"type":"content_start","contentType":"tool","toolName":"skills","input":{"skill":"serbian-voice"}}}',
    '{"type":"agent_event","event":{"type":"content_end","contentType":"tool","toolName":"skills","output":{"error":"denied"}}}',
    '{"type":"agent_event","event":{"type":"usage","cost":0.04718175,"totalCost":0.04718175}}',
    '{"type":"agent_event","event":{"type":"content_start","contentType":"text","text":"K"}}',
    '{"type":"agent_event","event":{"type":"content_end","contentType":"text","text":"Kafa je topao napitak koji mnogi ljudi piju ujutru."}}',
    '{"type":"agent_event","event":{"type":"usage","cost":0.005124525,"totalCost":0.052306275}}',
])


def test_parse_cline_extracts_text_excluding_reasoning_and_tool():
    text, _ = run_cli.parse_output("cline", CLINE_SAMPLE)
    assert text == "Kafa je topao napitak koji mnogi ljudi piju ujutru."   # not the reasoning/tool events


def test_parse_cline_cost_is_cumulative_last_wins_not_summed():
    _, cost = run_cli.parse_output("cline", CLINE_SAMPLE)
    assert abs(cost - 0.052306275) < 1e-9, "must keep the LAST cumulative totalCost, not sum per-event"


def test_parse_free_cli_short_circuits_zero_cost():
    text, cost = run_cli.parse_output("opencode", "  neki srpski tekst  ")
    assert text == "neki srpski tekst" and cost == 0.0        # free CLIs never parse/charge


def test_parse_cline_tolerates_nonjson_log_lines():
    noisy = "starting up...\n" + CLINE_SAMPLE + "\n[warn] trailing garbage"
    text, cost = run_cli.parse_output("cline", noisy)
    assert text == "Kafa je topao napitak koji mnogi ljudi piju ujutru."
    assert abs(cost - 0.052306275) < 1e-9


def test_spent_cost_sums_raw_and_errors_ledger(tmp_path):
    (tmp_path / "raw.jsonl").write_text(
        json.dumps({"id": "a", "cost": 0.01}) + "\n" + json.dumps({"id": "b", "cost": 0.02}) + "\n")
    (tmp_path / "errors.jsonl").write_text(          # failures are billed too, must count
        json.dumps({"id": "c", "cost": 0.03}) + "\n")
    assert abs(run_cli.spent_cost(tmp_path) - 0.06) < 1e-9


def test_spent_cost_missing_or_uncosted_lines_are_zero(tmp_path):
    assert run_cli.spent_cost(tmp_path) == 0.0               # no ledger yet
    (tmp_path / "raw.jsonl").write_text(json.dumps({"id": "a"}) + "\nnot json\n")   # no cost / junk
    assert run_cli.spent_cost(tmp_path) == 0.0


def test_failed_ids_skips_only_chronic_failures(tmp_path):
    lines = [json.dumps({"id": "x"}) for _ in range(3)] + [json.dumps({"id": "y"})]
    (tmp_path / "errors.jsonl").write_text("\n".join(lines) + "\n")
    assert run_cli.failed_ids(tmp_path, 3) == {"x"}          # x failed 3x -> skip; y once -> retry
    assert run_cli.failed_ids(tmp_path, 4) == set()          # threshold not reached


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    import tempfile
    fails = 0
    for name, fn in fns:
        try:
            if "tmp_path" in fn.__code__.co_varnames:
                with tempfile.TemporaryDirectory() as d:
                    fn(pathlib.Path(d))
            else:
                fn()
            print(f"PASS {name}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {name}: {e}")
    print(f"\n{len(fns) - fails}/{len(fns)} passed")
    raise SystemExit(1 if fails else 0)
