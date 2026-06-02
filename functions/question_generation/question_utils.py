SS_QUESTION_PROMPT = """
You are an expert benchmark designer creating memory-evaluation questions from a dialogue excerpt.

Your task is to generate {n} questions that test DETAILED knowledge of the following conversation excerpt.
Someone who has NOT read the excerpt must not be able to answer the questions correctly.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}
Session Date: {date}

===== DIALOGUE EXCERPT =====
{dialogue_excerpt}

===== GUIDELINES =====
1. Every question must be answerable SOLELY from the excerpt above.
2. Ask about specific, concrete details (named items, exact quantities, specific advice given, etc.).
3. Vary question style: what / who / where / how / which.
4. "answer" is an ordered list: the FIRST element is the gold (exact) answer; subsequent elements are
   gradient answers that are partially correct or plausible alternatives, earning decreasing partial
   credit.  Maximum 4 elements total.  Minimum 1 (use 1 element only for yes/no questions).
5. Generate a self-contained question with exactly one correct answer of 3 words or fewer, requiring no external knowledge.
6. Questions should contain assistant response queries and user information queries.
7. "supporting_facts" is the round identifier where the answer is found (e.g. "D1:3").
   If the answer spans multiple rounds use a comma-separated list (e.g. "D1:3,D1:4").
8. "label" must always equal answer[0]. "type" must be "ss".

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "What snacks did the user decide to buy before the next flight?",
    "answer": ["Nuts and yogurt", "Nuts", "Snacks"],
    "supporting_facts": "D1:8",
    "type": "ss",
    "label": "Nuts and yogurt"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

MS_QUESTION_PROMPT = """
You are an expert benchmark designer creating cross-session memory-evaluation questions.

Your task is to generate {n} questions that REQUIRE connecting or comparing information across the
multiple conversation sessions shown below.  A question that can be answered from a single session
alone is NOT acceptable.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

Timeline Overview:
{timeline_summary}

===== MULTI-SESSION DIALOGUE EXCERPTS =====
{dialogue_excerpt}

===== GUIDELINES =====
1. Every question must require facts from AT LEAST TWO different sessions.
2. Good question patterns:
   - Reject-and-endorse: "what option was rejected and what was endorsed?"
   - Cross-recap consistency: "what brand/value did the assistant standardise on?"
   - Unique-feature-across-sessions: "what was mentioned only in the discussion that did NOT name X?"
   - Conjunction across cases: "what two medications were used across these two cases?"
   - Repeated-recommendation: "what brand has the assistant recommended on more than one occasion?"
3. The supporting_facts MUST cite rounds from at least two distinct sessions (e.g. "D2:5,D16:9").
   If you can answer the question from one session alone, redesign — it is SS, not MS.
4. "answer" ordered list: first element is the gold answer, then gradient alternatives (up to 4 total).
5. "label" must equal answer[0].
6. Generate self-contained questions whose canonical answer is a short noun phrase.
7. "type" must be "ms".

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "For a budgie's home oxygen setup, what option has the user consistently rejected and what option has been endorsed?",
    "answer": [
      "Rejected the smart breeding box; endorsed the temperature and ventilation control system",
      "Rejected the breeding box; endorsed the ventilation system",
      "Ventilation control system"
    ],
    "supporting_facts": "D2:5,D16:9",
    "type": "ms",
    "label": "Rejected the smart breeding box; endorsed the temperature and ventilation control system"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

TR_QUESTION_PROMPT = """
You are an expert benchmark designer creating TEMPORAL REASONING evaluation questions.

Your task is to generate {n} questions that specifically test the ability to reason about TIME
from the conversation sessions below.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

Event Timeline (with dates):
{timeline_with_dates}

===== SESSION EXCERPTS (with session dates) =====
{dialogue_excerpt}

===== GUIDELINES =====
1. Every question must hinge on temporal information: specific dates, event ordering, durations,
   or time-range counts.
2. Use a mix of patterns:
   - Absolute-date lookup: "When did the user perform <event>? Return in YYYY-MM-DD."
   - Date → topic: "What did the user talk about on <date>?"
   - Bracket reasoning: "Between event X and event Y, what was the user treating?"
   - Before/after filter: "What was discussed AFTER <date>?"
   - Span: "Across what date range does the user's case log run?"
3. Use EXACT dates visible in the session headers or timeline; do not invent dates.
4. supporting_facts SHOULD cite both the event round AND any recap round that gives the absolute date,
   e.g. "D11:10,D16:12".  Without the recap round, dates are often unknowable from dialogue alone.
5. "answer" ordered list: gold first, then format variants ("YYYY-MM-DD", "Month D, YYYY", "Mon D YYYY")
   for date-typed answers; for date-→-topic items list 2–3 alternative correct topic phrases.
6. "label" must equal answer[0].  "type" must be "tr".

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "When did the user perform a dental cleaning on the senior cat? Return in the format YYYY-MM-DD.",
    "answer": ["2023-10-10", "October 10, 2023", "Oct 10 2023"],
    "supporting_facts": "D8:8,D9:3,D16:12",
    "type": "tr",
    "label": "2023-10-10"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

TH_QUESTION_PROMPT = """
You are an expert benchmark designer creating THEMATIC case-management questions.

Your task is to generate {n} questions that bundle MULTIPLE decisions or pieces of advice from
ONE case thread (one event, one topic) into a single comprehensive answer.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

===== DIALOGUE EXCERPT (single case thread) =====
{dialogue_excerpt}

===== GUIDELINES =====
1. The question must require BUNDLING multiple facts from one case (NOT a single lookup).
2. Use these patterns:
   - Diagnostic conclusion given symptoms + finding.
   - Step-by-step plan covering ≥ 3 stages (cleaning, bandaging, monitoring, follow-up …).
   - Diagnosis → next step (cause-and-action pair).
   - Technique application (full set of dos and don'ts in a procedure).
   - Decision-justification pair ("which sites were named and which is preferred").
   - Owner-instruction bundle ("what two adjustments should be recommended").
3. supporting_facts cites 1–2 rounds within the SAME case thread (often within one session).
   If the question asks across unrelated topics, that's MS, not TH — redesign.
4. "answer" is a 3-tier ordered list of decreasing specificity:
     index 0 — full canonical answer covering ALL key elements asked about.
     index 1 — short paraphrase keeping the main points.
     index 2 — single-keyword fallback (one noun phrase).
5. "label" must equal answer[0].  "type" must be "th".
6. Do not test trivial single-fact lookups; if the answer is a single named entity, this is SS, not TH.

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "For the dog deep-laceration case, what wound-care priorities did the assistant outline (cleaning, bandaging, monitoring, activity, follow-up)?",
    "answer": [
      "Clean with mild antiseptic or saline (avoid harsh chemicals); apply sterile bandage and change daily; monitor for redness, swelling, discharge, foul odor; restrict activity; schedule a follow-up to remove stitches",
      "Clean with antiseptic, daily bandage change, monitor infection, restrict activity, follow-up",
      "Clean and bandage"
    ],
    "supporting_facts": "D6:5",
    "type": "th",
    "label": "Clean with mild antiseptic or saline (avoid harsh chemicals); apply sterile bandage and change daily; monitor for redness, swelling, discharge, foul odor; restrict activity; schedule a follow-up to remove stitches"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

II_QUESTION_PROMPT = """
You are an expert benchmark designer creating IMPLICIT INFERENCE questions about the user.

Your task is to generate {n} questions whose answers are NEVER stated outright in the dialogue or
timeline, but can be inferred from at least TWO distinct clues across the corpus.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

Timeline Overview:
{timeline_summary}

===== DIALOGUE EXCERPTS (use these as clue sources) =====
{dialogue_excerpt}

===== GUIDELINES =====
1. The answer must NOT appear verbatim anywhere in the dialogue or timeline.
   If your candidate answer phrase appears literally in the corpus, change wording.
2. Each item must rest on ≥ 2 distinct clues (cite the rounds that contain those clues in supporting_facts).
3. Use these inference patterns:
   - Role / seniority beyond the named profession (e.g. "lead clinician" inferred from team-management cues).
   - Practice / domain mix inferred from the species/topics the user handles.
   - Hierarchy position: primary-care vs specialty (inferred from out-referrals).
   - After-hours / on-call coverage inferred from a single night-page event.
   - Clientele level (lay public vs professionals) inferred from explanation style.
   - Communication style (structured / documentation-oriented).
   - Personal possession (slipped-in "my cat / my apartment").
   - Daily pace, emotional toll, leadership posture, decision style.
   - Calendar season inferred from absolute dates without the season being named.
   - Belief about clients inferred from how the user explains things.
4. AVOID statistical / counting questions ("which species occurs most often"). The answer must be a
   short qualitative noun phrase, not a frequency.
5. "answer" is a 3-tier ordered list (full canonical, short paraphrase, single-keyword fallback).
6. "label" must equal answer[0].  "type" must be "ii".

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "From the clinic-management decisions the user makes (standardising supply brands across the team, training new staff, distributing reference cards), what role does the user most likely hold beyond simply being the named profession?",
    "answer": [
      "a senior / lead clinician",
      "lead clinician",
      "head veterinarian"
    ],
    "supporting_facts": "D12:6,D16:2,D16:3",
    "type": "ii",
    "label": "a senior / lead clinician"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

MR_QUESTION_PROMPT = """
You are an expert benchmark designer creating MULTIMODAL REASONING questions whose answers must be
read from images (charts, photos, product labels) attached to the cited dialogue rounds.

Each cited round has an associated image set (filenames listed in the IMAGE INVENTORY below).  Your
job is to write questions whose answers are visible in those images but NOT verbatim repeated in
the assistant's dialogue text.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

===== DIALOGUE EXCERPT (with image inventory per round) =====
{dialogue_excerpt}

===== IMAGE INVENTORY =====
{image_inventory}

===== GUIDELINES =====
1. Each question MUST be unanswerable from the dialogue text alone.  Before drafting, check that the
   numeric value, label text, or visual feature you are targeting does NOT appear verbatim in the
   assistant's reply at the cited round.
2. Use a mix of three sub-patterns:
   - chart lookup: a specific bar/line value or chart-metadata item (axis label, legend, source, n).
   - chart aggregate: max / min / average / difference across a clearly specified set of bars.
   - non-chart visual: object color, printed label text, body markings on a regular photo.
3. supporting_facts cites the round whose img_file array contains the image holding the answer.
4. "answer" is a 3-tier list including format variants for numeric answers
   (e.g. ["35%", "35", "35 percent"]).
5. "label" must equal answer[0].  "type" must be "mr".
6. If the targeted detail does appear in the assistant text, either pick a different fact from the
   image OR explicitly mark the question as a text-baseline item (rare; default to image-required).

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "In the political-affiliation chart, what percentage of Conservative respondents view the issue as 'Not a moral issue'?",
    "answer": ["21%", "21"],
    "supporting_facts": "D15:1",
    "type": "mr",
    "label": "21%"
  }},
  {{
    "question": "What colour is the stethoscope shown in one of the hamster-care images?",
    "answer": ["pink", "Pink"],
    "supporting_facts": "D8:2",
    "type": "mr",
    "label": "pink"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

FM_QUESTION_PROMPT = """
You are an expert benchmark designer creating FIND-MATCHING-IMAGE questions.  Given a brief
description, the model must identify which image (from the candidate set) is being described.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

===== DIALOGUE EXCERPT (with image inventory per round) =====
{dialogue_excerpt}

===== IMAGE INVENTORY =====
{image_inventory}

===== GUIDELINES =====
1. Pick a round whose img_file array contains 2–5 candidate images.  ONE of those images is the
   "label" — the image you describe.  The other images form the candidate set the model must
   choose from.
2. The description must distinguish the LABEL image from every OTHER candidate in the same round.
   If two candidates fit the description, redesign.
3. Stick to obvious, verifiable visual content (animal type, dominant colors, prominent objects,
   legible label text, room/equipment context).  Avoid invented details.
4. "answer" is the candidate-set list (label first, then the others).  Filenames use the
   convention "img_<number>" (no file extension).
5. "label" is the image filename of the unique correct match.
6. "type" must be "fm".  "supporting_facts" is the round id whose img_file array gave the candidate set.
7. End the question with: "Return the image file name in the format img_<number>."

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "Among the user's images, which image shows a tabby kitten lying on a green patterned blanket with a pink and yellow bandage and an IV line on its leg? Return the image file name in the format img_<number>.",
    "answer": ["img_25", "img_26", "img_27"],
    "supporting_facts": "D5:1",
    "type": "fm",
    "label": "img_25"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""

FJ_QUESTION_PROMPT = """
You are an expert benchmark designer creating 4-option MULTIPLE-CHOICE factual judgement
questions.  The model must output a single letter (A, B, C, or D).

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

===== DIALOGUE EXCERPT =====
{dialogue_excerpt}

===== GUIDELINES =====
1. Every item embeds the four options inside the question text on separate lines, like:
     <stem>
     (A) <option text>
     (B) <option text>
     (C) <option text>
     (D) <option text>
     Output only the letter of the correct option (A, B, C, or D).
   The final instruction line is REQUIRED verbatim so EM scoring works on a single letter.
2. Distractors must come from the SAME dialogue corpus (other entities, brands, drugs, devices,
   numbers the assistant mentioned in different contexts) — NOT plausible-sounding inventions.
   Distractors that can be eliminated by world knowledge alone are forbidden.
3. Distribute the correct letter so it does NOT favour any single position across this batch.
   Aim for roughly equal A/B/C/D across the {n} items.
4. supporting_facts cites the round(s) where the correct fact lives; multi-round items cite ≥ 2 rounds.
5. "answer" is a single-element list containing the correct letter.  "label" is the same letter.
6. "type" must be "fj".

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "Across the glove discussions and the standardisation recap, which two surgical-glove brands does the clinic stock?\\n(A) Cardinal and Halyard (latex)\\n(B) Sempermed and Curad (vinyl)\\n(C) Medline and Kimberly-Clark (latex)\\n(D) Ansell and microFlex (nitrile)\\nOutput only the letter of the correct option (A, B, C, or D).",
    "answer": ["D"],
    "supporting_facts": "D7:8,D8:1,D16:2",
    "type": "fj",
    "label": "D"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""


MR_PDF_QUESTION_PROMPT = """
You are an expert benchmark designer creating MULTIMODAL REASONING questions whose answers must be
read from the attached PDF document "{doc_title}" (filename: "{pdf_file}").

The attached images ARE the pages of this PDF (one image per page, in order). Treat them as the
real document: read all text, captions, equations, tables, code blocks, and figures.

===== CONTEXT =====
User Persona: {persona}
Core Event: {core_event}

===== DIALOGUE EXCERPT (rounds that referenced this PDF) =====
{dialogue_excerpt}

===== GUIDELINES =====
1. Every question MUST be answerable from the attached PDF pages alone.  Do NOT use outside
   knowledge.  Phrase questions so the answerer can find evidence on the attached pages.
2. Before drafting a question, check that the numeric value, label text, or visual feature you are
   targeting does NOT appear verbatim in the assistant's reply at the cited dialogue round — the
   answer must require reading the PDF, not parroting the assistant.
3. Use a mix of three sub-patterns:
   - direct lookup: a specific number / definition / claim / equation appearing on a page.
   - aggregate / comparison: contrast two values, identify max/min, compute a difference.
   - figure / table read: bar value, table cell, axis label, legend item, caption detail.
4. supporting_facts cites a dialogue round id (e.g. "D7:3") whose user record carries this PDF.
5. "answer" is a 3-tier list including format variants for numeric answers
   (e.g. ["35%", "35", "35 percent"]).
6. "label" must equal answer[0].  "type" must be "mr".

===== OUTPUT FORMAT =====
Return a valid JSON array. Example:

[
  {{
    "question": "In Table 2 of the lecture deck, what is the reported accuracy on the validation set?",
    "answer": ["0.87", "87%", "0.870"],
    "supporting_facts": "D7:3",
    "type": "mr",
    "label": "0.87"
  }}
]

Generate EXACTLY {n} questions. Return ONLY the JSON array — no markdown fences, no extra text.
"""


PROMPTS: dict[str, str] = {
    "SS": SS_QUESTION_PROMPT,
    "MS": MS_QUESTION_PROMPT,
    "TR": TR_QUESTION_PROMPT,
    "TH": TH_QUESTION_PROMPT,
    "II": II_QUESTION_PROMPT,
    "MR": MR_QUESTION_PROMPT,
    "FM": FM_QUESTION_PROMPT,
    "FJ": FJ_QUESTION_PROMPT,
}
