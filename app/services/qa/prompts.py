"""
Prompt content for the QA scoring task.

A 4B model with no fine-tuning leans heavily on few-shot examples for two
things: (1) staying inside the JSON schema in a way that *matches the
rubric*, not just valid JSON, and (2) calibrating what a "3" vs a "5" on
each dimension actually looks like. Three worked examples are included -
cancellation, rebooking, baggage - matching the categories requested for
this first pass. More can be appended here later without touching any
other file.
"""

SYSTEM_PROMPT = """You are a QA analyst for an airline customer support team. You review \
chat transcripts between a support agent and a passenger, and score the agent's handling \
of the conversation.

Score each dimension from 1 (poor) to 5 (excellent):
- empathy: did the agent acknowledge the passenger's situation/frustration appropriately?
- resolution_accuracy: was the outcome correct per airline policy (rebooking, refund, \
compensation eligibility, etc.)?
- policy_compliance: did the agent follow policy and flag anything requiring escalation \
(e.g. EU261 compensation for 3+ hour EU delays/cancellations, missed connections, force majeure)?
- communication_clarity: was the agent clear, concise, and easy to follow?
- efficiency: was the issue resolved without unnecessary back-and-forth?

Categorize the primary issue, note any secondary issues, and flag anything a manager should \
know about (e.g. "compliance_risk", "missed_escalation", "tone_issue", "missed_upsell", \
"possible_eu261_entitlement"). Give 1-3 concrete strengths and 1-3 concrete, actionable \
improvements the agent could make - written as coaching feedback, not generic praise/criticism.

Respond with ONLY the JSON object matching the given schema. No extra commentary, no markdown \
fences, no explanation of your reasoning."""


FEW_SHOT_EXAMPLES = [
    {
        "transcript": """[customer] My flight AA1042 to Chicago just got cancelled and nobody told me why. I have a meeting tomorrow morning.
[agent] I'm sorry about that, AA1042 was cancelled due to a mechanical issue reported this afternoon. Let me get you rebooked right away.
[agent] I have you on AA2210 tomorrow at 6:15am, arriving 8:40am - that should get you in well before a morning meeting. I've also added a $12 meal voucher for the inconvenience.
[customer] Ok that works, thank you.
[agent] You're welcome! Your new confirmation is AA2210-CHI9. Let me know if you need anything else.""",
        "output": {
            "category": "cancellation",
            "secondary_issues": ["rebooking"],
            "scores": {
                "empathy": 4,
                "resolution_accuracy": 5,
                "policy_compliance": 4,
                "communication_clarity": 5,
                "efficiency": 5,
            },
            "resolved": True,
            "escalation_needed": False,
            "flags": [],
            "strengths": [
                "Rebooked proactively onto a flight that meets the passenger's stated deadline",
                "Gave a specific new confirmation number so the passenger has a clear next step",
            ],
            "improvements": [
                "Could have explained why a mechanical cancellation doesn't qualify for cash compensation, to preempt a follow-up question",
            ],
            "summary": "Domestic mechanical cancellation, agent rebooked promptly onto a flight meeting the passenger's deadline and added a goodwill voucher.",
        },
    },
    {
        "transcript": """[customer] My connection in Frankfurt got in late and I missed my flight to Delhi, LH760. What do I do now??
[agent] I understand, let me look into it.
[agent] I see LH760 has already departed. There is another flight LH762 tonight but it's full.
[customer] So what am I supposed to do, sleep at the airport??
[agent] I can put you on standby for LH762.
[customer] Standby isn't a real answer. I have connecting family waiting in Delhi.
[agent] I can also offer tomorrow morning's flight, confirmed seat, plus a hotel voucher for tonight since this was a missed connection caused by our delayed inbound flight.
[customer] Fine, that's better. Book that one.
[agent] Done, you're confirmed on LH764 tomorrow 7:05am with a hotel voucher already sent to your email.""",
        "output": {
            "category": "missed_connection",
            "secondary_issues": ["rebooking"],
            "scores": {
                "empathy": 2,
                "resolution_accuracy": 4,
                "policy_compliance": 3,
                "communication_clarity": 3,
                "efficiency": 2,
            },
            "resolved": True,
            "escalation_needed": False,
            "flags": ["missed_escalation", "tone_issue"],
            "improvements": [
                "Lead with the confirmed-seat-plus-hotel option immediately - offering standby first as the only option needlessly escalated the passenger's frustration",
                "Acknowledge the inconvenience explicitly before presenting options, especially once the passenger raises waiting family",
            ],
            "strengths": [
                "Correctly identified this as an airline-caused missed connection and issued a hotel voucher without being asked",
            ],
            "summary": "Missed connection due to a delayed inbound flight; agent reached the right outcome but only after the passenger pushed back on an initial standby-only offer.",
        },
    },
    {
        "transcript": """[customer] My bag didn't arrive on flight BA118 into London. I also have a small dog traveling in cargo and I need to confirm she's ok.
[agent] I'm very sorry - let's sort out both. For the bag, can you give me the baggage tag number?
[customer] Its 0125-884213
[agent] Thanks, I've filed a report, reference BAG-LHR-77210. It's showing as still in Frankfurt and should arrive on the next BA flight tonight, and we'll deliver it to your address at no cost.
[customer] Ok. And my dog?
[agent] Let me check with cargo handling on that.
[agent] Your dog was confirmed offloaded and collected from the pet relief area, she's fine and with ground staff now.
[customer] Thank god. That's all I needed.""",
        "output": {
            "category": "baggage",
            "secondary_issues": ["ancillary_or_pet"],
            "scores": {
                "empathy": 4,
                "resolution_accuracy": 4,
                "policy_compliance": 3,
                "communication_clarity": 4,
                "efficiency": 4,
            },
            "resolved": True,
            "escalation_needed": False,
            "flags": ["missed_upsell"],
            "strengths": [
                "Got a specific baggage reference number and delivery commitment in one exchange",
                "Proactively followed up on the pet welfare question rather than treating it as an afterthought",
            ],
            "improvements": [
                "Should have proactively mentioned compensation for essential items (toiletries etc.) for the delayed bag, rather than waiting to be asked",
            ],
            "summary": "Delayed bag plus an in-cargo pet on the same flight; agent resolved both with concrete references, but missed mentioning essential-item compensation for the delay.",
        },
    },
]


def build_few_shot_block() -> str:
    import json

    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        blocks.append(
            f"Example {i} transcript:\n{ex['transcript']}\n\n"
            f"Example {i} correct output:\n{json.dumps(ex['output'])}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(transcript_text: str) -> str:
    return (
        f"{build_few_shot_block()}\n\n---\n\n"
        f"Now score this transcript the same way:\n\n{transcript_text}"
    )
