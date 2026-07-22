"""
Deterministic (non-LLM) timing checks for chat transcripts.

Both checks below are pure Python - no Ollama call - because what they
measure is exact arithmetic on known timestamps plus simple phrase
matching, not judgment. That keeps the LLM's job scoped to what actually
needs a model (empathy, clarity, policy calls) and keeps these checks fast,
free, and 100% reproducible, per the same "don't trust the model with
arithmetic" principle `qa_service._overall_score` already follows.

Both checks require `ChatMessage.elapsed_seconds` (seconds since the start
of the chat) on every message. Real transcripts pulled from a live chat
system would need to supply this from actual send timestamps; if it's
missing anywhere, both checks report `evaluated=False` rather than
guessing.

## Hold-time compliance
The Hold Time Policy has two independent rules, both checked here:

Rule 1 - Hold Duration (what the agent SAYS). Per policy, whenever an agent
places the passenger on hold, the *stated* duration must always be exactly
5 minutes. "Please allow me 1 minute" / "2 minutes" / "3 minutes" /
"4 minutes" / "10 minutes" are all violations of this rule - only stating
"5 minutes" is compliant. This is checked purely against what the agent
said, independent of how the hold actually plays out.

Rule 2 - Hold Resumption (what the agent DOES). The agent must actually
return within 5 minutes (300s) of announcing the hold - this is the one
fixed benchmark that matters here, always, regardless of what the agent
said out loud. Coming back sooner than 5 minutes is good (better CSAT) and
never flagged. Taking longer than 5 minutes is the only violation
condition for this rule. We do NOT compare actual time against whatever
duration the agent happened to state ("3 minutes", "5 minutes", etc.) -
only against the fixed policy benchmark.

These two rules are evaluated independently per hold - a hold can violate
Rule 1 only, Rule 2 only, both, or neither. Multiple holds in one
transcript are each judged independently with their own fresh 5-minute
timer (per policy Rule 3 - "each hold starts a completely new 5-minute
timer").

## Idle-protocol adherence
Per the documented "near-perfect chat" flow, if the passenger goes quiet:
  - ~2 min idle  -> agent sends a first check-in
  - ~3 min idle, still no response -> agent sends a final message and
    closes the chat (rather than going silent or just disappearing)
These two checkpoints - 2 minutes and 3 minutes - are fixed by the product
spec, not configurable per transcript.

Importantly, if the passenger's silence was because the AGENT placed them on
hold to do their own work, that hold time is not passenger-idle time. The
2/3-minute clock only starts once the agent resumes with a real update and is
now the one waiting on the passenger - a "still there?" ping is for when the
*passenger* goes quiet, not for the agent to announce their own return from a
hold. So when a hold precedes the quiet stretch, the check-in/final-notice
checkpoints are measured from the agent's resumption message, not from the
customer's last message or the hold announcement.
"""
import re

from app.models.qa_schemas import (
    ChatTranscript,
    HoldCheck,
    HoldTimeCompliance,
    IdleProtocolCompliance,
    IdleWindowCheck,
)

# Fixed company hold benchmark for Rule 2 (actual resumption time) - the
# ONLY number that matters for that rule, regardless of what the agent states.
HOLD_POLICY_SECONDS = 300  # 5 min

# Rule 1: the stated hold duration itself must always be exactly this - not
# "anything <= 5 minutes", not "close to 5 minutes". Only this exact value
# is compliant; every other stated duration is a violation of Rule 1.
REQUIRED_STATED_HOLD_SECONDS = 300  # 5 min

# Idle-protocol checkpoints (fixed by spec - see module docstring).
IDLE_FIRST_CHECKIN_SECONDS = 120  # 2 min
IDLE_FINAL_NOTICE_SECONDS = 180  # 3 min
IDLE_TOLERANCE_SECONDS = 45  # grace window either side of each checkpoint

_DURATION_RE = re.compile(r"(\d+)\s*(?:more\s+)?min(?:ute)?s?", re.IGNORECASE)
_CHECKIN_PHRASES = ("checking in", "still there")
_FINAL_NOTICE_PHRASES = (
    "go ahead and close",
    "close this chat",
    "close this for now",
    "close the chat",
)


def _has_timestamps(transcript: ChatTranscript) -> bool:
    return bool(transcript.messages) and all(m.elapsed_seconds is not None for m in transcript.messages)


def _is_checkin(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in _CHECKIN_PHRASES)


def _is_final_notice(text: str) -> bool:
    low = text.lower()
    return any(phrase in low for phrase in _FINAL_NOTICE_PHRASES)


def _is_hold_announcement(text: str) -> bool:
    return bool(_DURATION_RE.search(text))


def check_hold_time_compliance(transcript: ChatTranscript) -> HoldTimeCompliance:
    if not _has_timestamps(transcript):
        return HoldTimeCompliance(
            evaluated=False,
            note="Transcript is missing per-message timestamps (elapsed_seconds) - hold-time check skipped.",
        )

    messages = transcript.messages
    holds: list[HoldCheck] = []

    for i, msg in enumerate(messages):
        if msg.speaker != "agent":
            continue
        match = _DURATION_RE.search(msg.text)
        if not match:
            continue

        stated_seconds = int(match.group(1)) * 60

        # Rule 1: the stated duration must be exactly 5 minutes, full stop.
        # This is independent of everything below - it's checked purely
        # against what the agent said, not against what happens afterward.
        stated_duration_compliant = stated_seconds == REQUIRED_STATED_HOLD_SECONDS

        # Find the next agent message that actually delivers something,
        # skipping pure check-in pings (which don't resolve the hold).
        resolution_elapsed = None
        for j in range(i + 1, len(messages)):
            follow_up = messages[j]
            if follow_up.speaker != "agent" or _is_checkin(follow_up.text):
                continue
            resolution_elapsed = follow_up.elapsed_seconds
            break

        if resolution_elapsed is None:
            # Hold was announced but never followed up on in this transcript -
            # nothing to measure Rule 2 (resumption time) against, so (as
            # before) we don't report this as a hold at all.
            continue

        actual_seconds = resolution_elapsed - msg.elapsed_seconds
        # Rule 2: compliance is against the fixed company benchmark, NOT the
        # stated duration - see module docstring. This is evaluated
        # independently of Rule 1 above; either can fail on its own.
        overage = actual_seconds - HOLD_POLICY_SECONDS
        resumption_exceeded = overage > 0

        violations: list[str] = []
        if not stated_duration_compliant:
            violations.append("stated_duration_not_5_minutes")
        if resumption_exceeded:
            violations.append("resumption_exceeded_policy")

        holds.append(
            HoldCheck(
                agent_message_index=i,
                stated_text=msg.text,
                stated_seconds=stated_seconds,
                stated_duration_compliant=stated_duration_compliant,
                actual_seconds=actual_seconds,
                policy_seconds=HOLD_POLICY_SECONDS,
                exceeded=resumption_exceeded,
                overage_seconds=overage,
                violations=violations,
            )
        )

    return HoldTimeCompliance(
        evaluated=True,
        holds=holds,
        any_exceeded=any(h.exceeded for h in holds),
        any_violation=any(h.violations for h in holds),
        note=None if holds else "No stated hold/wait duration found in the transcript.",
    )


def check_idle_protocol_compliance(transcript: ChatTranscript) -> IdleProtocolCompliance:
    if not _has_timestamps(transcript):
        return IdleProtocolCompliance(
            evaluated=False,
            note="Transcript is missing per-message timestamps (elapsed_seconds) - idle-protocol check skipped.",
        )

    messages = transcript.messages
    n = len(messages)
    windows: list[IdleWindowCheck] = []

    i = 0
    while i < n:
        if messages[i].speaker != "customer":
            i += 1
            continue

        # Collect the run of agent-only messages immediately following this
        # customer message - i.e. the stretch where the passenger is quiet.
        j = i + 1
        agent_run = []
        while j < n and messages[j].speaker == "agent":
            agent_run.append(j)
            j += 1

        if not agent_run:
            i += 1
            continue

        customer_responded = j < n

        # Figure out where the check-in clock actually starts. If the agent
        # was working a hold they placed the passenger on, everything up to
        # their resumption message is hold-work time, not passenger-idle
        # time - that part is already covered by the hold-time check above.
        # The clock only starts once the agent hands control back with a
        # real update (a message that isn't itself a check-in/final-notice
        # ping) and is now the one waiting on the passenger.
        first_ping_idx = next(
            (k for k in agent_run if _is_checkin(messages[k].text) or _is_final_notice(messages[k].text)),
            None,
        )
        pre_ping = agent_run if first_ping_idx is None else [k for k in agent_run if k < first_ping_idx]

        preceded_by_hold_announcement = (
            i > 0 and messages[i - 1].speaker == "agent" and _is_hold_announcement(messages[i - 1].text)
        )
        run_resumes_from_hold = bool(pre_ping) and (len(pre_ping) < len(agent_run) or preceded_by_hold_announcement)

        if run_resumes_from_hold:
            wait_start_index = pre_ping[-1]
            wait_start = messages[wait_start_index].elapsed_seconds
            trailing = [] if first_ping_idx is None else [k for k in agent_run if k >= first_ping_idx]
            if not trailing:
                # The agent resumed with an update but nothing after it
                # (customer responded right away, or the transcript just
                # ends there) - no check-in/final-notice timing to evaluate.
                i = j
                continue
        else:
            wait_start_index = i
            wait_start = messages[i].elapsed_seconds
            trailing = agent_run

        idle_end = messages[j].elapsed_seconds if customer_responded else messages[agent_run[-1]].elapsed_seconds
        idle_duration = idle_end - wait_start

        # Only windows long enough to plausibly need the protocol are evaluated.
        if idle_duration >= IDLE_FIRST_CHECKIN_SECONDS - IDLE_TOLERANCE_SECONDS:
            windows.append(
                _evaluate_idle_window(messages, wait_start_index, wait_start, trailing, customer_responded, idle_duration)
            )

        i = j

    return IdleProtocolCompliance(
        evaluated=True,
        windows=windows,
        any_violation=any(w.violations for w in windows),
        note=None if windows else "No idle window long enough to require the check-in protocol was found.",
    )


def _evaluate_idle_window(messages, wait_start_index, wait_start, trailing, customer_responded, idle_duration):
    violations: list[str] = []

    # Prefer an explicit "checking in" style ping; if none exists, fall back
    # to the agent's first message in the trailing stretch - some response is
    # still a response, just not phrased as a check-in.
    checkin_idx = next((k for k in trailing if _is_checkin(messages[k].text)), trailing[0])
    first_checkin_seconds = messages[checkin_idx].elapsed_seconds - wait_start
    first_checkin_on_time = abs(first_checkin_seconds - IDLE_FIRST_CHECKIN_SECONDS) <= IDLE_TOLERANCE_SECONDS

    if not first_checkin_on_time:
        violations.append("checkin_early" if first_checkin_seconds < IDLE_FIRST_CHECKIN_SECONDS else "checkin_late")

    final_idx = next((k for k in trailing if _is_final_notice(messages[k].text)), None)
    final_notice_seconds = messages[final_idx].elapsed_seconds - wait_start if final_idx is not None else None
    final_notice_on_time = (
        abs(final_notice_seconds - IDLE_FINAL_NOTICE_SECONDS) <= IDLE_TOLERANCE_SECONDS
        if final_notice_seconds is not None
        else None
    )

    if customer_responded:
        outcome = "customer_responded"
    elif final_idx is not None:
        outcome = "closed_after_final_notice"
        if not final_notice_on_time:
            violations.append("final_notice_early" if final_notice_seconds < IDLE_FINAL_NOTICE_SECONDS else "final_notice_late")
    else:
        outcome = "no_final_notice_given"
        if idle_duration >= IDLE_FINAL_NOTICE_SECONDS - IDLE_TOLERANCE_SECONDS:
            # Passenger was quiet long enough that a final notice/close
            # should have happened by now, and it didn't - the agent
            # appears to have just gone silent instead of closing the loop.
            violations.append("missing_final_notice")

    return IdleWindowCheck(
        wait_start_index=wait_start_index,
        idle_duration_seconds=idle_duration,
        customer_responded=customer_responded,
        first_checkin_seconds=first_checkin_seconds,
        first_checkin_on_time=first_checkin_on_time,
        final_notice_sent=final_idx is not None,
        final_notice_seconds=final_notice_seconds,
        final_notice_on_time=final_notice_on_time,
        outcome=outcome,
        violations=violations,
    )
