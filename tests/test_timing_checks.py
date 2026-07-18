"""
Tests for the deterministic (non-LLM) hold-time and idle-protocol checks.

These are pure functions over a ChatTranscript, so no Ollama mocking is
needed here - unlike test_qa_api.py, which exercises the full /analyze
endpoint with the LLM call monkeypatched.
"""
from app.models.qa_schemas import ChatMessage, ChatTranscript
from app.services.qa import timing_checks


def _transcript(messages):
    return ChatTranscript(transcript_id="T", messages=messages)


def _m(speaker, text, t=None):
    return ChatMessage(speaker=speaker, text=text, elapsed_seconds=t)


# ---------------------------------------------------------------- hold time

def test_hold_time_skipped_without_timestamps():
    tr = _transcript([_m("agent", "give me about 5 minutes")])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.evaluated is False
    assert result.holds == []


def test_hold_time_within_stated_duration_not_flagged():
    tr = _transcript([
        _m("agent", "could I place you on hold, about 5 minutes?", 0),
        _m("customer", "sure", 5),
        _m("agent", "thanks for waiting, here's your update", 280),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.evaluated is True
    assert len(result.holds) == 1
    assert result.holds[0].exceeded is False
    assert result.any_exceeded is False


def test_hold_time_exceeded_is_soft_flagged():
    tr = _transcript([
        _m("agent", "brief hold, about 3 minutes", 0),
        _m("customer", "ok", 5),
        _m("agent", "thanks for waiting, here's your update", 600),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.any_exceeded is True
    assert result.holds[0].stated_seconds == 180
    assert result.holds[0].overage_seconds == 420


def test_hold_time_check_in_does_not_count_as_resolution():
    tr = _transcript([
        _m("agent", "hold on, about 2 minutes", 0),
        _m("agent", "just checking in - are you still there?", 90),
        _m("agent", "thanks for waiting, all done", 150),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert len(result.holds) == 1
    # Resolution should be the 3rd message (150s), not the check-in at 90s -
    # and since 150s > the stated 120s, this is (correctly) a soft flag.
    assert result.holds[0].actual_seconds == 150
    assert result.holds[0].exceeded is True


def test_hold_time_no_duration_stated():
    tr = _transcript([
        _m("agent", "let me take a look", 0),
        _m("customer", "ok", 5),
        _m("agent", "all set", 30),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.evaluated is True
    assert result.holds == []
    assert result.note is not None


# ------------------------------------------------------------- idle protocol

def test_idle_protocol_skipped_without_timestamps():
    tr = _transcript([_m("customer", "hi")])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert result.evaluated is False


def test_idle_protocol_on_time_checkin_then_customer_returns():
    tr = _transcript([
        _m("customer", "sure, go ahead", 0),
        _m("agent", "just checking in - are you still there?", 120),
        _m("customer", "yes still here", 130),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert result.evaluated is True
    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.outcome == "customer_responded"
    assert window.first_checkin_on_time is True
    assert window.violations == []


def test_idle_protocol_late_checkin_is_flagged():
    tr = _transcript([
        _m("customer", "sure, go ahead", 0),
        _m("agent", "still working on it, thanks for your patience", 500),
        _m("customer", "hello?", 520),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    window = result.windows[0]
    assert "checkin_late" in window.violations


def test_idle_protocol_final_notice_on_time_no_response():
    tr = _transcript([
        _m("customer", "go ahead", 0),
        _m("agent", "just checking in - are you still there?", 120),
        _m("agent", "I'll go ahead and close this chat for now - reach back out anytime!", 190),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    window = result.windows[0]
    assert window.customer_responded is False
    assert window.outcome == "closed_after_final_notice"
    assert window.violations == []


def test_idle_protocol_missing_final_notice_is_flagged():
    tr = _transcript([
        _m("customer", "go ahead", 0),
        _m("agent", "just checking in - are you still there?", 120),
        _m("agent", "still nothing back from them, moving to another chat", 300),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    window = result.windows[0]
    assert window.outcome == "no_final_notice_given"
    assert "missing_final_notice" in window.violations


def test_idle_protocol_ignores_short_gaps():
    tr = _transcript([
        _m("customer", "hi", 0),
        _m("agent", "sure, one sec", 5),
        _m("customer", "thanks", 10),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert result.evaluated is True
    assert result.windows == []
