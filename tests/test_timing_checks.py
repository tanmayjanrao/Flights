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


def test_hold_time_within_policy_not_flagged():
    # Stated duration is deliberately much shorter than the actual time -
    # compliance is judged against the fixed 300s policy, not what the agent
    # said, so this should still pass since 280s < 300s.
    tr = _transcript([
        _m("agent", "could I place you on hold, about 1 minute?", 0),
        _m("customer", "sure", 5),
        _m("agent", "thanks for waiting, here's your update", 280),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.evaluated is True
    assert len(result.holds) == 1
    assert result.holds[0].policy_seconds == 300
    assert result.holds[0].exceeded is False
    assert result.any_exceeded is False


def test_hold_time_exceeded_fixed_policy_is_soft_flagged():
    # Stated duration is deliberately much longer than the fixed policy -
    # compliance still goes against the fixed 300s benchmark, not the 10
    # minutes the agent stated, so this should still be flagged.
    tr = _transcript([
        _m("agent", "brief hold, about 10 minutes", 0),
        _m("customer", "ok", 5),
        _m("agent", "thanks for waiting, here's your update", 600),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.any_exceeded is True
    assert result.holds[0].stated_seconds == 600  # kept for context only, not used for exceeded
    assert result.holds[0].policy_seconds == 300
    assert result.holds[0].overage_seconds == 300


def test_hold_time_check_in_does_not_count_as_resolution():
    tr = _transcript([
        _m("agent", "hold on, about 2 minutes", 0),
        _m("agent", "just checking in - are you still there?", 90),
        _m("agent", "thanks for waiting, all done", 150),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert len(result.holds) == 1
    # Resolution should be the 3rd message (150s), not the check-in at 90s -
    # and since 150s is within the fixed 300s policy benchmark, this is not
    # flagged (even though it's past the stated 120s).
    assert result.holds[0].actual_seconds == 150
    assert result.holds[0].exceeded is False


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


# ---------------------------------------- idle protocol - resumption-aware


def test_idle_protocol_hold_work_time_is_not_idle_time():
    # The agent announces a long hold, the passenger acks it, and the agent
    # takes a while to come back - but comes back with a genuine update
    # (not a check-in ping) and the passenger replies right away. This is
    # hold-work time (covered separately by the hold-time check), not
    # passenger-idle time, so it must NOT produce an idle-protocol window at
    # all - there's nothing to check in on.
    tr = _transcript([
        _m("agent", "could I place you on hold, about 5 minutes?", 0),
        _m("customer", "sure, no problem", 5),
        _m("agent", "I'm back - here's your update", 290),
        _m("customer", "great, thanks!", 300),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert result.evaluated is True
    assert result.windows == []


def test_idle_protocol_checkin_clock_starts_at_resumption_not_hold_announcement():
    # The agent holds, resumes with a real update, and ONLY THEN does the
    # passenger go quiet long enough to need a check-in. The check-in is on
    # time relative to the resumption (2 min after it), even though it's far
    # more than 2 min after the original hold announcement - proving the
    # clock starts at the resumption, not the hold announcement.
    tr = _transcript([
        _m("agent", "could I place you on hold, about 5 minutes?", 0),
        _m("customer", "sure, go ahead", 5),
        _m("agent", "I'm back - here's your update, let me know what you'd like to do", 290),
        _m("agent", "just checking in - are you still there?", 410),  # 120s after resumption (290), not after hold announce (0)
        _m("customer", "sorry, yes still here", 420),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert result.evaluated is True
    assert len(result.windows) == 1
    window = result.windows[0]
    assert window.outcome == "customer_responded"
    assert window.first_checkin_seconds == 120
    assert window.first_checkin_on_time is True
    assert window.violations == []


def test_idle_protocol_still_flags_genuinely_slow_agent_when_no_hold_was_announced():
    # No hold was ever announced here - the agent just goes quiet for a long
    # time with no interstitial update. This should be treated the same as
    # before the rework: the clock starts from the passenger's last message.
    tr = _transcript([
        _m("customer", "sure, go ahead", 0),
        _m("agent", "still working on it, thanks for your patience", 500),
        _m("customer", "hello?", 520),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    window = result.windows[0]
    assert "checkin_late" in window.violations
