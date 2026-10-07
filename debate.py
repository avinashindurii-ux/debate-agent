"""
Multi-Agent Debate System (v8)
--------------------------------
Three agents debate a topic:
  - Agent A (Proponent): argues FOR the topic
  - Agent B (Opponent): argues AGAINST the topic
  - Agent C (Moderator): tracks repetition/dodging, flags unsupported claims
    every round, detects sub-topic narrowing, and issues direct instructions

Changes from v7, based on review:

  ISSUE 6 - the v7 "verdict" (judging which agent argued more effectively)
  structurally favored whoever spoke second in each round, since engagement
  scoring naturally rewards having the last word. Rather than try to correct
  for that bias in scoring logic, the verdict mechanism is removed entirely.

  The long post-debate moderator report (arguments raised by A/B, direct
  clashes, dropped arguments, unsupported-claims recap) is also gone - it
  was essay-length, duplicated what the listener already heard, and doesn't
  fit a spoken/TTS product.

  In its place: a short, neutral, spoken-length closing recap (3-5
  sentences, same plain-language and no-markdown rules as every other turn)
  that just restates each side's core final position. No winner, no
  comparative judgment, nothing structured.

Uses the Groq API (OpenAI-compatible chat completions).
"""

import os
import re
import sys
import time
from groq import Groq, APIError, RateLimitError

# ----------------------------
# Config
# ----------------------------
MODEL = "openai/gpt-oss-120b"
ROUNDS = 4
MAX_RETRIES = 3
RETRY_DELAY = 5
MAX_WORDS_PER_TURN = 90
MAX_WORDS_CLOSING = 130
NARROWING_OVERLAP_THRESHOLD = 0.35
NARROWING_STREAK_LIMIT = 2
DIMENSION_STALL_LIMIT = 2
DIMENSION_TOUCH_THRESHOLD = 0.15

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "with", "that", "this", "these", "those", "is", "are",
    "was", "were", "be", "been", "being", "to", "of", "in", "on", "for", "as", "by", "at", "it",
    "its", "their", "they", "he", "she", "his", "her", "them", "both", "agent", "agents", "while",
    "argued", "countered", "debated", "claiming", "asserting", "emphasizing", "reiterating",
    "against", "would", "could", "should", "not", "no", "than", "then", "so", "if", "each", "such",
}

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ----------------------------
# Helpers
# ----------------------------
def call_model(system_prompt, user_prompt, temperature=0.9):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError:
            wait = RETRY_DELAY * attempt
            print(f"[rate limited, retrying in {wait}s...]")
            time.sleep(wait)
        except APIError as e:
            print(f"[API error: {e}, retrying...]")
            time.sleep(RETRY_DELAY)
    raise RuntimeError("Failed to get a response after retries.")


def strip_markdown(text):
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.M)
    return text.strip()


def word_count(text):
    return len(text.split())


def get_spoken_text(system_prompt, user_prompt, temperature=0.9, max_words=MAX_WORDS_PER_TURN):
    """Call the model, strip markdown, retry once if too long for speech, then hard-truncate as a last resort."""
    text = strip_markdown(call_model(system_prompt, user_prompt, temperature))
    if word_count(text) > max_words:
        stricter_prompt = (
            system_prompt
            + f"\n\nSTRICT REMINDER: your last attempt ran over {max_words} words. Keep it under "
            f"{int(max_words * 0.75)} words this time, cut to the essentials."
        )
        text = strip_markdown(call_model(stricter_prompt, user_prompt, temperature))
        if word_count(text) > max_words:
            words = text.split()
            trimmed = " ".join(words[:max_words])
            last_period = trimmed.rfind(".")
            text = trimmed[:last_period + 1] if last_period != -1 else trimmed
    return text


def format_transcript(transcript):
    return "\n\n".join(f"{t['speaker']}: {t['text']}" for t in transcript)


def significant_words(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 3}


def overlap_ratio(text_a, text_b):
    words_a, words_b = significant_words(text_a), significant_words(text_b)
    if not words_a or not words_b:
        return 0.0
    shared = words_a & words_b
    smaller = min(len(words_a), len(words_b))
    return len(shared) / smaller if smaller else 0.0


def dimension_touched(dimension, round_text):
    dim_words = significant_words(dimension)
    if not dim_words:
        return False
    text_words = significant_words(round_text)
    shared = dim_words & text_words
    return (len(shared) / len(dim_words)) >= DIMENSION_TOUCH_THRESHOLD


# ----------------------------
# Argument dimensions (hidden coverage tracker, never shown to the user)
# ----------------------------
def dimension_generator_prompt(topic):
    return (
        f"For a debate on: \"{topic}\", list 6-8 distinct argument dimensions or angles, tailored "
        "specifically to this topic, not generic filler (e.g. enforcement feasibility, equity, cost, "
        "precedent, individual autonomy, long-term effects).\n"
        "Respond with ONLY a plain list, one short dimension name per line, no numbering, no explanation."
    )


def generate_dimensions(topic):
    result = call_model(
        "You help structure debates by identifying distinct argument dimensions.",
        dimension_generator_prompt(topic),
        temperature=0.7,
    )
    return [line.strip("-• ").strip() for line in result.splitlines() if line.strip()]


# ----------------------------
# Shared rules
# ----------------------------
SHARED_RULES = """
You are speaking out loud in a live debate, not writing an essay.

Hard rules, no exceptions:
1. Respond in 2-4 natural spoken sentences, roughly 40-70 words total. No bullet points, no numbered
   lists, no markdown headers, no bold or italic formatting. Speak the way a person actually talks.
2. NO FABRICATED CITATIONS. Never invent a named study, institution, or precise statistic. If unsure,
   hedge in plain speech: "reports suggest," "from what I've seen," "in practice." Honest vagueness
   beats fake precision.
3. If your opponent's last turn raised a specific point, respond to it by name or keyword before
   adding anything new. Don't just change the subject.
4. AVOID JARGON. Don't use technical, legal, or specialist terms unless you immediately explain them
   in one plain phrase a general listener would understand.
5. DON'T TURN YOUR POSITION INTO A TAGLINE. If you're about to restate a stance you've raised before,
   only include it if you add a genuinely new reason or angle. If you have nothing new to add, leave
   it out this round.
6. Do not open with "You're right that..." or any fixed phrase.
7. No stage directions, no meta-commentary about debating. Just talk.

How to handle the moderator's last note about you (if any is given below): choose ONE of three modes
based on what it actually said, don't default to the same one every time.
- Brief acceptance: only if the moderator said you dodged, repeated, or made an unsupported claim.
  One short natural sentence acknowledging it, varied wording, then move on.
- Pushback: if the moderator flagged something you can actually defend, push back briefly.
- No acknowledgment: the most common choice. If the note doesn't concern you, just debate normally.
"""


def proponent_prompt(topic):
    return f"You are Agent A, arguing FOR the topic: \"{topic}\".\n{SHARED_RULES}"


def opponent_prompt(topic):
    return f"You are Agent B, arguing AGAINST the topic: \"{topic}\".\n{SHARED_RULES}"


def moderator_prompt(topic):
    return (
        f"You are Agent C, the Moderator of a spoken debate on: \"{topic}\".\n"
        "Review the most recent ROUND: Agent A's latest turn followed by Agent B's latest turn.\n"
        "Judge:\n"
        "1. Did each agent respond to the opponent's specific last point, or dodge by changing subject?\n"
        "2. Did either agent repeat an argument or tagline they already used without a new angle?\n"
        "3. Did either agent cite a specific stat, named study, or named institution unhedged? Quote "
        "the exact claim if so.\n\n"
        "Respond in this exact format, nothing else:\n"
        "STATUS_A: on-topic | dodged | repeated\n"
        "STATUS_B: on-topic | dodged | repeated\n"
        "UNSUPPORTED_CLAIMS: <quote the exact unhedged claim if any, else 'none'>\n"
        "SUMMARY: <one plain sentence capturing what this round covered, for use as memory in later "
        "rounds, not a verdict, just what was argued>\n"
        "INSTRUCTION_FOR_NEXT_ROUND: <a direct instruction if intervention is needed, naming the "
        "specific point dodged or repeated, else 'none'>"
    )


def closing_recap_prompt(topic):
    return (
        f"You are the Moderator, closing a spoken debate on: \"{topic}\".\n"
        "Give a short, neutral, spoken-style wrap-up, 3-5 natural sentences, roughly 80-130 words "
        "total. No markdown, no bullet points, no headers, no sections.\n\n"
        "Just restate each side's core final position in plain language, e.g. what Agent A's central "
        "case was, what Agent B's central case was, and optionally one honest line about what both "
        "sides touched on.\n\n"
        "Do NOT declare a winner. Do NOT say which agent argued better, was more convincing, or held "
        "up better. Do NOT list arguments, clashes, dropped points, or unsupported claims. This is a "
        "plain recap for a listener, not a report, read it the way a person would wrap up a "
        "conversation out loud."
    )


# ----------------------------
# Moderator check
# ----------------------------
def moderator_check(topic, transcript):
    prompt_text = (
        f"Full transcript so far:\n\n{format_transcript(transcript)}\n\n"
        "Evaluate the most recent round only (Agent A's last turn and Agent B's last turn)."
    )
    result = call_model(moderator_prompt(topic), prompt_text, temperature=0.2)

    status_a, status_b = "on-topic", "on-topic"
    unsupported, summary, instruction = "none", "", "none"
    for line in result.splitlines():
        upper = line.strip().upper()
        if upper.startswith("STATUS_A:"):
            status_a = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("STATUS_B:"):
            status_b = line.split(":", 1)[1].strip().lower()
        elif upper.startswith("UNSUPPORTED_CLAIMS:"):
            unsupported = line.split(":", 1)[1].strip()
        elif upper.startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()
        elif upper.startswith("INSTRUCTION_FOR_NEXT_ROUND:"):
            instruction = line.split(":", 1)[1].strip()

    return status_a, status_b, unsupported, summary, instruction


def build_compressed_context(round_summaries, opponent_last_turn, opponent_label):
    parts = []
    if round_summaries:
        parts.append("Debate so far, round by round:\n" + "\n".join(
            f"Round {i+1}: {s}" for i, s in enumerate(round_summaries)
        ))
    else:
        parts.append("(debate just started)")
    if opponent_last_turn:
        parts.append(f"\n{opponent_label}'s most recent turn, in full:\n{opponent_last_turn}")
    return "\n\n".join(parts)


def build_redirect_note(topic, untouched_dims):
    dim_text = (
        f"raise or revisit one of these angles you haven't used yet: {', '.join(untouched_dims)}"
        if untouched_dims else
        "revisit an angle from earlier in the debate that hasn't been developed"
    )
    return (
        "\n\n[Moderator: this debate has narrowed into the same back-and-forth for too long. For this "
        f"round, {dim_text}. Return to the core question: \"{topic}\", in plain terms. Do NOT introduce "
        "any new unverified facts, statistics, or named examples while redirecting, reason from the "
        "angle itself or restate the core stakes plainly instead.]"
    )


def build_soft_nudge(dimension):
    return (
        f"\n\n[One angle that hasn't come up yet: {dimension}. Consider raising it if it genuinely "
        "strengthens your point, but don't force it if you have a stronger argument to make.]"
    )


# ----------------------------
# Main debate loop
# ----------------------------
def _console(kind, d):
    """Default event handler: prints the debate to the terminal."""
    if kind == "start":
        print(f"\n{'='*60}\nDEBATE TOPIC: {d['topic']}\n{'='*60}\n")
    elif kind == "round":
        print(f"\n--- Round {d['round']} ---\n")
    elif kind == "turn":
        print(f"{d['speaker']}: {d['text']}\n")
    elif kind == "moderator":
        flag_text = f" ({', '.join(d['flags'])})" if d["flags"] else ""
        print(f"[Moderator{flag_text}: {d['summary']}]")
        if d["unsupported"].lower() != "none":
            print(f"[Unsupported claim flagged: {d['unsupported']}]")
    elif kind == "redirect":
        print(f"[Moderator: sub-topic has narrowed for {d['streak']} consecutive rounds, "
              "redirecting next round]")
    elif kind == "recap":
        print(f"\n{'='*60}\nCLOSING RECAP\n{'='*60}\n\n{d['text']}")


def run_debate(topic, rounds=ROUNDS, on_event=None):
    """Run the debate. on_event(kind, data) receives progress events; defaults to terminal output."""
    emit = on_event or _console
    full_transcript = []
    round_summaries = []
    last_a_turn = None
    last_b_turn = None
    instruction = None
    narrowing_streak = 0

    emit("start", {"topic": topic, "rounds": rounds})
    all_dimensions = generate_dimensions(topic)
    touched = {d: False for d in all_dimensions}
    rounds_since_new_dimension = 0

    for round_num in range(1, rounds + 1):
        emit("round", {"round": round_num})

        redirecting = narrowing_streak >= NARROWING_STREAK_LIMIT
        untouched = [d for d, was_touched in touched.items() if not was_touched]
        nudging = (not redirecting) and untouched and rounds_since_new_dimension >= DIMENSION_STALL_LIMIT

        # Agent A
        context = build_compressed_context(round_summaries, last_b_turn, "Agent B")
        user_prompt = f"Topic: {topic}\n\n{context}"
        if instruction and instruction != "none":
            user_prompt += f"\n\n[Moderator's note last round: {instruction}]"
        if redirecting:
            user_prompt += build_redirect_note(topic, untouched)
        elif nudging:
            user_prompt += build_soft_nudge(untouched[0])
        a_text = get_spoken_text(proponent_prompt(topic), user_prompt)
        full_transcript.append({"speaker": "Agent A (For)", "text": a_text})
        last_a_turn = a_text
        emit("turn", {"side": "A", "speaker": "Agent A (For)", "text": a_text})

        # Agent B
        context = build_compressed_context(round_summaries, last_a_turn, "Agent A")
        user_prompt = f"Topic: {topic}\n\n{context}"
        if instruction and instruction != "none":
            user_prompt += f"\n\n[Moderator's note last round: {instruction}]"
        if redirecting:
            user_prompt += build_redirect_note(topic, untouched)
        elif nudging:
            user_prompt += build_soft_nudge(untouched[0])
        b_text = get_spoken_text(opponent_prompt(topic), user_prompt)
        full_transcript.append({"speaker": "Agent B (Against)", "text": b_text})
        last_b_turn = b_text
        emit("turn", {"side": "B", "speaker": "Agent B (Against)", "text": b_text})

        if redirecting:
            narrowing_streak = 0

        # Update dimension coverage based on what actually got said this round
        new_dimension_touched = False
        combined_text = a_text + " " + b_text
        for dim in all_dimensions:
            if not touched[dim] and dimension_touched(dim, combined_text):
                touched[dim] = True
                new_dimension_touched = True
        rounds_since_new_dimension = 0 if new_dimension_touched else rounds_since_new_dimension + 1

        # Moderator reviews the full round
        status_a, status_b, unsupported, summary, next_instruction = moderator_check(topic, full_transcript)
        flags = []
        if status_a != "on-topic":
            flags.append(f"A: {status_a}")
        if status_b != "on-topic":
            flags.append(f"B: {status_b}")
        emit("moderator", {"round": round_num, "flags": flags, "summary": summary,
                           "unsupported": unsupported})

        if round_summaries:
            overlap = overlap_ratio(summary, round_summaries[-1])
            narrowing_streak = narrowing_streak + 1 if overlap >= NARROWING_OVERLAP_THRESHOLD else 0

        if narrowing_streak >= NARROWING_STREAK_LIMIT:
            emit("redirect", {"streak": narrowing_streak})

        round_summaries.append(summary if summary else f"Round {round_num} exchange on {topic}.")
        instruction = next_instruction if next_instruction != "none" else None

    # Closing recap: short, neutral, spoken-length. No verdict, no report.
    context = format_transcript(full_transcript)
    recap = get_spoken_text(
        closing_recap_prompt(topic),
        f"Full transcript:\n\n{context}",
        temperature=0.3,
        max_words=MAX_WORDS_CLOSING,
    )
    emit("recap", {"text": recap})

    return full_transcript, recap


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("Error: set your GROQ_API_KEY environment variable first.")
        print('  set GROQ_API_KEY=your-key-here      (Windows)')
        print('  export GROQ_API_KEY="your-key-here"  (Mac/Linux)')
        sys.exit(1)

    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        topic = input("Enter a debate topic: ").strip()

    if not topic:
        print("No topic provided. Exiting.")
        sys.exit(1)

    run_debate(topic)
