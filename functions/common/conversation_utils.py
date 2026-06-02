TIMELINE_GENERATION_PROMPT = (
    "You are generating a timeline of events for a synthetic user's life.\n"
    "Return ONLY a JSON array. No markdown. No trailing commentary.\n\n"
    "Persona:\n%s\n\n"
    "Core event:\n%s\n\n"
    "Hard requirements:\n"
    "- Output EXACTLY %d event objects (no more, no fewer).\n"
    "- Every string field below must be NON-EMPTY. Never use \"\" or omit a key.\n"
    "- Chronological Event_Time across the story.\n\n"
    "Each event MUST be an object with these keys IN THIS ORDER:\n"
    '  1) "Event_Index": integer, 1 through N, matching array position (first object = 1, last = N).\n'
    '  2) "Event_Description": one clear sentence, third person, what happens in the user\'s life.\n'
    '  3) "Query_Description": one sentence for multimodal dialogue. Use the PROTAGONIST name from the Persona '
    '(the name right after "Your name is ..."). Pattern: '
    "[That name] shares photos (or screenshots) of [concrete visual things] and asks [one specific question]. "
    "Do not use Carl or any other name unless the Persona literally names that person. "
    'Example: "Nora shares photos of a water quality log sheet and asks how to interpret ammonia versus last week."\n'
    '  4) "Event_Time": string YYYY-MM-DD.\n'
    '  5) "Keyword": short English phrase for image search (about 2–6 words).\n'
)

DISTRACTOR_TIMELINE_PROMPT = (
    "You are generating DISTRACTOR life events for the same synthetic user, to "
    "be interleaved with their main story. These events must look like ordinary "
    "moments in this person's life but must NOT advance, mirror, or reference "
    "the core event arc.\n"
    "Return ONLY a JSON array. No markdown. No trailing commentary.\n\n"
    "Persona:\n%s\n\n"
    "Core event (DO NOT touch these themes):\n%s\n\n"
    "Existing main-line events for chronological context (date range only — do "
    "not duplicate their topics):\n%s\n\n"
    "Hard requirements:\n"
    "- Output EXACTLY %d event objects.\n"
    "- Every string field must be NON-EMPTY. Never use \"\" or omit a key.\n"
    "- Each event must be a believable, mundane moment for this persona that is "
    "ORTHOGONAL to the core event (different domain: food / errands / hobbies / "
    "minor health / weather / family chat / commute / app glitch / etc.).\n"
    "- Event_Time values must fall WITHIN the date range of the main-line events "
    "above and be plausibly interleaved (do not bunch on one date).\n\n"
    "Each event MUST be an object with these keys IN THIS ORDER:\n"
    '  1) "Event_Description": one clear sentence, third person, an off-topic moment in the user\'s life.\n'
    '  2) "Query_Description": one sentence in the form '
    "[Protagonist name from the Persona] shares photos (or screenshots) of [concrete off-topic visual] "
    "and asks [one specific question]. The visual and the question must be unrelated to the core event.\n"
    '  3) "Event_Time": string YYYY-MM-DD inside the main timeline\'s date span.\n'
    '  4) "Keyword": short English phrase for image search (about 2–6 words), about the OFF-topic content.\n'
)

TIMELINE_SELFCHECK_PROMPT = (
    "You are auditing a generated timeline against the persona and the core event.\n"
    "Return ONLY a JSON object. No markdown. No commentary outside the object.\n\n"
    "Persona:\n%s\n\n"
    "Core event:\n%s\n\n"
    "Timeline (JSON array of event objects):\n%s\n\n"
    "Check three things and report each finding with the offending Event_Index "
    "values (1-based, matching the timeline above):\n"
    "  1) persona_violations  — events that contradict the persona (wrong name, "
    "     wrong profession, behaviours impossible for this persona, etc.).\n"
    "  2) core_event_violations — events that fail to express or that contradict "
    "     the core_event arc (NOTE: events explicitly marked off-topic are NOT "
    "     violations — only flag main-line events that drop the arc).\n"
    "  3) contradictions — pairs of events that contradict each other "
    "     (impossible date order, mutually exclusive states, name drift between "
    "     events, an item said to be lost and then used as if owned, etc.).\n\n"
    "Output shape (use empty arrays when nothing is wrong):\n"
    "{\n"
    '  "persona_violations":    [{"event_index": <int>, "issue": "<short reason>"}],\n'
    '  "core_event_violations": [{"event_index": <int>, "issue": "<short reason>"}],\n'
    '  "contradictions":        [{"event_indices": [<int>, <int>], "issue": "<short reason>"}],\n'
    '  "ok": <true if all three arrays are empty, else false>\n'
    "}\n"
)

TIMELINE_REPAIR_PROMPT = (
    "You previously generated this timeline:\n%s\n\n"
    "An audit found these issues:\n%s\n\n"
    "Persona:\n%s\n\n"
    "Core event:\n%s\n\n"
    "Rewrite ONLY the events flagged by the audit. Leave every other event "
    "byte-identical (same keys, same values, same order). For pair-wise "
    "contradictions, pick whichever event the audit blames and edit just that "
    "one — do not touch both unless both indices appear in the report.\n\n"
    "Return ONLY the full repaired timeline as a JSON array, same shape and "
    "length as the input. No markdown. No commentary.\n"
)


DIALOGUE_SELFCHECK_CHUNK_PROMPT = (
    "You are auditing a contiguous slice of a generated dialogue against the "
    "user's persona and the core event of their story.\n"
    "Return ONLY a JSON object. No markdown. No commentary outside the object.\n\n"
    "Persona:\n%s\n\n"
    "Core event:\n%s\n\n"
    "Timeline overview (for context; events may already be consumed):\n%s\n\n"
    "Dialogue rounds in this chunk (each has a unique round id like D1:7):\n%s\n\n"
    "Audit FOUR categories and report each finding with the offending round id "
    "(the User/Assistant turn id you saw above):\n"
    "  1) persona_violations  — the user or assistant says/does something that "
    "     contradicts the persona (wrong name, wrong profession, knowledge or "
    "     gear the persona could not plausibly have, etc.).\n"
    "  2) core_event_violations — main-line rounds that fail to advance, mirror, "
    "     or relate to the core_event arc.  Rounds for off-topic moments are "
    "     fine; only flag drift in rounds clearly meant to be part of the arc.\n"
    "  3) hallucinations — the assistant invents entities / data / brands / "
    "     people / numbers that the persona, the core_event, the timeline, or "
    "     earlier rounds never grounded.\n"
    "  4) internal_contradictions — within this chunk only, two rounds that "
    "     contradict each other (a fact stated in one is denied or violated in "
    "     another).\n\n"
    "Output shape (use empty arrays when nothing is wrong):\n"
    "{\n"
    '  "persona_violations":       [{"round": "<id>", "issue": "<short reason>"}],\n'
    '  "core_event_violations":    [{"round": "<id>", "issue": "<short reason>"}],\n'
    '  "hallucinations":           [{"round": "<id>", "issue": "<short reason>"}],\n'
    '  "internal_contradictions":  [{"rounds": ["<id>", "<id>"], "issue": "<short reason>"}],\n'
    '  "ok": <true if all four arrays are empty, else false>\n'
    "}\n"
)


DIALOGUE_SELFCHECK_CROSS_PROMPT = (
    "You are doing a CROSS-CHUNK pass over a long generated dialogue, looking "
    "for contradictions that span chunks (not visible inside any single chunk).\n"
    "Return ONLY a JSON object. No markdown.\n\n"
    "Persona:\n%s\n\n"
    "Core event:\n%s\n\n"
    "Per-chunk findings (already collected; the cross pass should ADD to these, "
    "not duplicate them):\n%s\n\n"
    "Dialogue overview — every round's user-line summarized in one short line "
    "with its round id:\n%s\n\n"
    "Flag cross-chunk problems where one round contradicts a much earlier or "
    "much later round, or where a persona/core_event invariant fails when read "
    "end-to-end.  Do NOT repeat findings already listed in the per-chunk "
    "report; only report NEW issues that require comparing distant rounds.\n\n"
    "Output shape (use empty arrays when nothing is wrong):\n"
    "{\n"
    '  "cross_chunk_contradictions": [{"rounds": ["<id>", "<id>"], "issue": "<short reason>"}],\n'
    '  "global_persona_violations":  [{"round": "<id>", "issue": "<short reason>"}],\n'
    '  "global_core_event_violations": [{"round": "<id>", "issue": "<short reason>"}],\n'
    '  "ok": <true if all three arrays are empty, else false>\n'
    "}\n"
)


DIALOGUE_REPAIR_PROMPT = (
    "You are rewriting ONE flagged round in a generated dialogue so it no "
    "longer triggers the listed issues, while keeping the conversation natural "
    "and consistent with the surrounding rounds.\n"
    "Return ONLY a JSON object.  No markdown.  No commentary outside the object.\n\n"
    "Persona:\n%s\n\n"
    "Core event:\n%s\n\n"
    "Neighbouring rounds (sliding window — rounds BEFORE the target, then the "
    "TARGET round itself, then rounds AFTER; preserve their text and intent):\n%s\n\n"
    "Issues reported against the target round:\n%s\n\n"
    "Rewrite the TARGET round so that:\n"
    "- It still advances the same beat the original round was attempting "
    "  (don't change the topic unless an issue says you must).\n"
    "- It no longer triggers any of the reported issues.\n"
    "- It reads naturally given the preceding rounds and sets up the following "
    "  rounds without contradicting them.\n"
    "- Keep the same image / pdf attachments — do NOT introduce new ones.\n\n"
    "Output shape:\n"
    "{\n"
    '  "user":      "<the new user message text>",\n'
    '  "assistant": "<the new assistant reply text>"\n'
    "}\n"
)


_DIALOGUE_SUMMARY_EXAMPLES = (
    "Examples:\n"
    "GOOD summary: 'The user asked about why two items differ; the assistant explained the contrast shown in the image.'\n"
    "BAD summary: 'They discussed stock market trends.' (only if the exchange was unrelated to stocks)\n"
    "BAD summary: Adds new numbers that neither message stated.\n"
)

DIALOGUE_SUMMARY_PROMPT = (
    "Summarize the following USER+ASSISTANT exchange in 1-2 short sentences.\n"
    "Stay faithful to what was actually said; do not add facts or numbers not in the exchange.\n"
    "Paraphrase at the level of **topics and numbers**; do not rely on file names in the summary.\n\n"
    + _DIALOGUE_SUMMARY_EXAMPLES
    + "\nReturn ONLY a flat JSON object with exactly one top-level string field named \"summary\". "
    "Do NOT nest another object inside \"summary\". Example: {\"summary\": \"One or two sentences here.\"}\n"
    "No markdown, no other keys.\n\n"
    "Exchange:\n%s\n"
)


THEMATIC_USER_PROMPT_INIT = (
    "You are the USER in a roleplay.\n"
    "Persona:\n%s\n\n"
    "Current event (JSON):\n%s\n\n"
    "You see the attached images (%s).\n"
    "Write ONE user message grounded in the event and images.\n"
    "Return ONLY JSON object with keys: speaker, timestamp, clean_text, image_description.\n"
    'speaker must be "User". timestamp should be a date string.\n'
)

THEMATIC_USER_PROMPT = (
    "You are the USER in a roleplay.\n"
    "Persona:\n%s\n\n"
    "Relevant event(s) (JSON):\n%s\n\n"
    "Conversation summaries so far (JSON):\n%s\n\n"
    "You see the attached images (%s).\n"
    "Write TWO consecutive user messages that progress the conversation.\n"
    "Return ONLY a JSON array of TWO objects, each with keys: speaker, timestamp, clean_text, image_description.\n"
    'speaker must be "User". timestamp should be a date string.\n'
)

THEMATIC_AGENT_PROMPT_INIT = (
    "You are the ASSISTANT.\n"
    "User said:\n%s\n\n"
    "You see the attached images (%s).\n"
    "Reply helpfully.\n"
    "Return ONLY JSON object with keys: speaker, clean_text.\n"
    'speaker must be "Agent".\n'
)

THEMATIC_AGENT_PROMPT = (
    "You are the ASSISTANT.\n"
    "Users said (JSON array of 2 user messages):\n%s\n\n"
    "You see the attached images (%s).\n"
    "Reply to each user message in order with TWO assistant messages.\n"
    "Return ONLY a JSON array of TWO objects, each with keys: speaker, clean_text.\n"
)


PDF_USER_PROMPT_INIT = """\
You are simulating a curious learner who has just opened the PDF document
"{doc_title}" for the first time. The document is most likely an academic
paper or a university lecture slide deck.

You can see the FIRST {n_pages} page(s) of this document as image(s) attached
below. Treat them as the actual pages: read all text, captions, equations,
tables, code blocks, and figures.

TASK
====
This is the FIRST PDF turn of the session, so start "clean_text" with a brief
natural opener that introduces the document — something like
"Hey, this is the slide deck for a class I've been taking recently — could
you help me work through it? I had a question about the first part." (or
similar; vary the wording, do not copy this verbatim).

Right after the opener, ask ONE focused, content-grounded question about
what is on these pages.

The QUESTION part MUST be specific:
  - about a definition, claim, dataset, equation, table cell, or figure;
  - never generic ("what is this paper about?", "summarize this");
  - never about the author / title / venue (that is metadata, not content).

OUTPUT FORMAT (single JSON object, no markdown fences):
{{
  "speaker":          "User",
  "clean_text":       "<intro sentence + the question, plain English, <= 110 words>",
  "pdf_description":  ["<<= 25-word summary of page 1>", ... exactly {n_pages} entries]
}}

JSON-safety: do NOT put raw LaTeX commands (e.g. \\(, \\sigma) inside string
values. Either spell math out in words, or escape every backslash as \\\\.

Extra constraints:
{extra}
"""

PDF_USER_PROMPT = """\
You are continuing a learning conversation about the PDF document
"{doc_title}". So far you have read {revealed} of {total} page(s).

Earlier dialogue summaries (most recent last):
{summaries}

You have just been shown {n_new} ADDITIONAL page(s). Read them carefully
(text, captions, equations, tables, code, figures).

TASK
====
Ask ONE focused, content-grounded question that builds on the conversation
so far OR that opens a clearly different angle drawn from the new pages.
Be specific and concrete — quote a number, label, or term from the pages.
Avoid trivial recap ("what does this say?").

OUTPUT FORMAT (single JSON object, no markdown fences):
{{
  "speaker":          "User",
  "clean_text":       "<the question, plain English, <= 80 words>",
  "pdf_description":  ["<<= 25-word summary of NEW page 1>", ... exactly {n_new} entries]
}}

JSON-safety: do NOT put raw LaTeX commands (e.g. \\(, \\sigma) inside string
values. Either spell math out in words, or escape every backslash as \\\\.

Extra constraints:
{extra}
"""

PDF_FOLLOWUP_USER_PROMPT = """\
You are still reading "{doc_title}". Your previous question was answered
by the assistant with:

\"\"\"{last_assistant}\"\"\"

TASK
====
Ask ONE follow-up question that probes deeper into what the assistant just
said. Do NOT introduce a brand-new topic. Pick something the assistant
claimed, asserted, glossed over, or hand-waved and ask for clarification,
derivation, evidence, mechanism, an edge case, a limitation, or an
implication. Be concise and specific.

OUTPUT FORMAT (single JSON object, no markdown fences):
{{
  "speaker":     "User",
  "clean_text":  "<follow-up question, <= 60 words>"
}}

Do NOT include pdf_description (no new pages were revealed for this turn).
"""

PDF_AGENT_PROMPT = """\
You are a knowledgeable teaching assistant helping a curious user understand
the PDF document "{doc_title}" (an academic paper or lecture slide deck).

You can see {revealed} page image(s) of the document attached below.

The user asks: "{user_text}"

Answer the question using ONLY information visible in the supplied page
images. If the answer is not on the visible pages, SAY SO honestly and point
to what IS on these pages that is most relevant. Quote numbers, equations,
column names, or figure labels verbatim where useful. Aim for 60-180 words.
Be precise, not flowery. No markdown headings.

OUTPUT FORMAT (single JSON object, no markdown fences):
{{
  "speaker":    "Agent",
  "clean_text": "<your answer in plain text>"
}}

JSON-safety: do NOT put raw LaTeX commands (e.g. \\(, \\sigma, \\sum) inside
"clean_text". Either describe equations in words ("sigma sub i", "P of s"),
use Unicode (sigma, pi), or escape every backslash as \\\\ — otherwise the JSON
is invalid.
"""
