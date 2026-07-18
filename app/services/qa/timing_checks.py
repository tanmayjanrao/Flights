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
Soft flag: does the agent's actual follow-up time exceed the duration they
told the passenger? Exceeding a stated hold time is common in practice
(see project handoff notes) so this is informational, not an auto-fail.

## Idle-protocol adherence
Per the documented "near-perfect chat" flow, if the passenger goes quiet:
  - ~2 min idle  -> agent sends a first check-in
  - ~3 min idle, still no response -> agent sends a final message and
    closes the chat (rather than going silent or just disappearing)
These two checkpoints - 2 minutes and 3 minutes - are fixed by the product
spec, not configurable per transcript.
"""
import re

from app.models.qa_schemas import (
    ChatTranscript,
    HoldCheck,
    HoldTimeCompliance,
    IdleProtocolCompliance,
    IdleWindowCheck,
)

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
            # nothing to measure it against.
            continue

        actual_seconds = resolution_elapsed - msg.elapsed_seconds
        overage = actual_seconds - stated_seconds
        holds.append(
            HoldCheck(
                agent_message_index=i,
                stated_text=msg.text,
                stated_seconds=stated_seconds,
                actual_seconds=actual_seconds,
                exceeded=overage > 0,
                overage_seconds=overage,
            )
        )

    return HoldTimeCompliance(
        evaluated=True,
        holds=holds,
        any_exceeded=any(h.exceeded for h in holds),
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

        idle_start = messages[i].elapsed_seconds
        customer_responded = j < n
        idle_end = messages[j].elapsed_seconds if customer_responded else messages[agent_run[-1]].elapsed_seconds
        idle_duration = idle_end - idle_start

        # Only windows long enough to plausibly need the protocol are evaluated.
        if idle_duration >= IDLE_FIRST_CHECKIN_SECONDS - IDLE_TOLERANCE_SECONDS:
            windows.append(_evaluate_idle_window(messages, i, idle_start, agent_run, customer_responded, idle_duration))

        i = j

    return IdleProtocolCompliance(
        evaluated=True,
        windows=windows,
        any_violation=any(w.violations for w in windows),
        note=None if windows else "No idle window long enough to require the check-in protocol was found.",
    )


def _evaluate_idle_window(messages, idle_start_index, idle_start, agent_run, customer_responded, idle_duration):
    violations: list[str] = []

    # Prefer an explicit "checking in" style ping; if none exists, fall back
    # to the agent's first message in the run - some response is still a
    # response, just not phrased as a check-in.
    checkin_idx = next((k for k in agent_run if _is_checkin(messages[k].text)), agent_run[0])
    first_checkin_seconds = messages[checkin_idx].elapsed_seconds - idle_start
    first_checkin_on_time = abs(first_checkin_seconds - IDLE_FIRST_CHECKIN_SECONDS) <= IDLE_TOLERANCE_SECONDS

    if not first_checkin_on_time:
        violations.append("checkin_early" if first_checkin_seconds < IDLE_FIRST_CHECKIN_SECONDS else "checkin_late")

    final_idx = next((k for k in agent_run if _is_final_notice(messages[k].text)), None)
    final_notice_seconds = messages[final_idx].elapsed_seconds - idle_start if final_idx is not None else None
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
        idle_start_index=idle_start_index,
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
