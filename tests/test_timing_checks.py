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


def test_hold_time_within_policy_not_flagged_for_resumption_but_stated_duration_still_wrong():
    # Rule 2 (resumption time, 280s < 300s) passes. But Rule 1 (stated
    # duration) requires the agent to have said exactly 5 minutes - "about 1
    # minute" is a Rule 1 violation on its own, independent of how quickly
    # the agent actually came back.
    tr = _transcript([
        _m("agent", "could I place you on hold, about 1 minute?", 0),
        _m("customer", "sure", 5),
        _m("agent", "thanks for waiting, here's your update", 280),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.evaluated is True
    assert len(result.holds) == 1
    assert result.holds[0].policy_seconds == 300
    assert result.holds[0].exceeded is False  # Rule 2: resumption on time
    assert result.holds[0].stated_duration_compliant is False  # Rule 1: didn't say 5 minutes
    assert result.holds[0].violations == ["stated_duration_not_5_minutes"]
    assert result.any_exceeded is False
    assert result.any_violation is True


def test_hold_time_exceeded_fixed_policy_is_soft_flagged():
    # Stated duration is deliberately long ("10 minutes") - this fails BOTH
    # rules independently: Rule 1 because it isn't 5 minutes, and Rule 2
    # because compliance still goes against the fixed 300s benchmark, not
    # the 10 minutes the agent stated, and 600s > 300s.
    tr = _transcript([
        _m("agent", "brief hold, about 10 minutes", 0),
        _m("customer", "ok", 5),
        _m("agent", "thanks for waiting, here's your update", 600),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.any_exceeded is True
    assert result.any_violation is True
    assert result.holds[0].stated_seconds == 600  # not used to decide `exceeded`, only Rule 1
    assert result.holds[0].stated_duration_compliant is False
    assert result.holds[0].policy_seconds == 300
    assert result.holds[0].overage_seconds == 300
    assert set(result.holds[0].violations) == {"stated_duration_not_5_minutes", "resumption_exceeded_policy"}


def test_hold_time_check_in_does_not_count_as_resolution():
    tr = _transcript([
        _m("agent", "hold on, about 2 minutes", 0),
        _m("agent", "just checking in - are you still there?", 90),
        _m("agent", "thanks for waiting, all done", 150),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert len(result.holds) == 1
    # Resolution should be the 3rd message (150s), not the check-in at 90s -
    # and since 150s is within the fixed 300s policy benchmark, Rule 2 is not
    # flagged (even though it's past the stated 120s). Rule 1 is still
    # flagged since "2 minutes" isn't the required 5 minutes.
    assert result.holds[0].actual_seconds == 150
    assert result.holds[0].exceeded is False
    assert result.holds[0].stated_duration_compliant is False
    assert result.holds[0].violations == ["stated_duration_not_5_minutes"]


# ------------------------------------------------- hold time - Rule 1 (stated duration)


def test_hold_time_stated_exactly_5_minutes_is_compliant():
    tr = _transcript([
        _m("agent", "Please allow me 5 minutes.", 0),
        _m("customer", "ok", 5),
        _m("agent", "thanks for waiting, here's your update", 200),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert result.holds[0].stated_duration_compliant is True
    assert result.holds[0].violations == []
    assert result.any_violation is False


def test_hold_time_stated_5_minutes_but_returns_late_is_rule_2_violation_only():
    # Agent said the compliant "5 minutes" (Rule 1 passes) but actually took
    # 6 minutes to come back (Rule 2 fails) - the two rules are independent,
    # so only Rule 2 should be flagged here.
    tr = _transcript([
        _m("agent", "Kindly allow me 5 minutes while I check this.", 0),
        _m("customer", "sure", 5),
        _m("agent", "thanks for waiting, here's your update", 360),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    hold = result.holds[0]
    assert hold.stated_duration_compliant is True
    assert hold.exceeded is True
    assert hold.violations == ["resumption_exceeded_policy"]


def test_hold_time_multiple_holds_are_independent():
    # Rule 3: each hold gets a brand new, independently-evaluated 5-minute
    # timer. First hold: compliant on both rules. Second hold: violates
    # both. Neither hold's evaluation should be affected by the other.
    tr = _transcript([
        _m("agent", "Please allow me 5 minutes.", 0),
        _m("customer", "ok", 5),
        _m("agent", "thanks for waiting, here's your update", 200),
        _m("customer", "one more thing", 210),
        _m("agent", "sure, please allow me 2 minutes for that", 215),
        _m("agent", "all set now, thanks for waiting", 900),
    ])
    result = timing_checks.check_hold_time_compliance(tr)
    assert len(result.holds) == 2

    first, second = result.holds
    assert first.stated_duration_compliant is True
    assert first.exceeded is False
    assert first.violations == []

    assert second.stated_duration_compliant is False
    assert second.exceeded is True
    assert set(second.violations) == {"stated_duration_not_5_minutes", "resumption_exceeded_policy"}

    assert result.any_violation is True


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
    # Per policy the final warning is due at a TOTAL of 5 minutes of idle
    # time (2 min first warning + a further 3 min of continued silence) -
    # not 3 minutes from idle start.
    tr = _transcript([
        _m("customer", "go ahead", 0),
        _m("agent", "just checking in - are you still there?", 120),
        _m("agent", "I'll go ahead and close this chat for now - reach back out anytime!", 295),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    window = result.windows[0]
    assert window.customer_responded is False
    assert window.outcome == "closed_after_final_notice"
    assert window.violations == []


def test_idle_protocol_final_notice_at_3_minutes_is_now_flagged_early():
    # This is the bug fix itself: a final warning fired at a 3-minute total
    # (the old, wrong benchmark) is really 2 minutes early against the real
    # 5-minute-total policy, and must be flagged as such, not treated as
    # compliant.
    tr = _transcript([
        _m("customer", "go ahead", 0),
        _m("agent", "just checking in - are you still there?", 120),
        _m("agent", "I'll go ahead and close this chat for now - reach back out anytime!", 190),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    window = result.windows[0]
    assert "final_notice_early" in window.violations


# ---------------------------------------- idle protocol - Rule 5 (reply cancels idle)


def test_idle_protocol_flags_warning_sent_after_customer_already_replied():
    # Mirrors the policy doc's own example: first warning at 2 min, customer
    # replies at 4:45, but the agent sends the final warning (and would then
    # disconnect) at 4:46 anyway - a "major QA violation" per policy, and one
    # the old length-gated implementation silently dropped and never flagged
    # at all because the resulting window is very short.
    tr = _transcript([
        _m("customer", "still need help with my booking", 0),
        _m("agent", "just checking in - are you still there?", 120),
        _m("customer", "sorry, yes - here's my confirmation", 285),
        _m("agent", "as we have not received a response, we will now close this chat", 286),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert result.any_violation is True
    violation_windows = [w for w in result.windows if "warning_sent_after_customer_reply" in w.violations]
    assert len(violation_windows) == 1
    assert violation_windows[0].outcome == "warning_sent_after_customer_reply"


def test_idle_protocol_first_warning_sent_just_before_customer_reply_is_not_a_reply_violation():
    # Contrast case: the first warning goes out BEFORE the customer's reply
    # (both are part of the normal, compliant flow) - the customer replying
    # shortly after a checkin is not itself a Rule 5 violation.
    tr = _transcript([
        _m("customer", "still there?", 0),
        _m("agent", "just checking in - are you still there?", 118),
        _m("customer", "yes, sorry", 130),
    ])
    result = timing_checks.check_idle_protocol_compliance(tr)
    assert all("warning_sent_after_customer_reply" not in w.violations for w in result.windows)


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
