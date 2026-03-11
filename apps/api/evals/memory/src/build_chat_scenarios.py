"""Build 30 chat extraction scenarios for the memory eval system."""

from __future__ import annotations

from pathlib import Path

from .scenarios import (
    ChatExchange,
    ChatExtractionScenario,
    ExpectedChatFact,
    save_chat_scenarios,
)


# ---------------------------------------------------------------------------
# A. Per-category extraction (12 scenarios)
# ---------------------------------------------------------------------------

def _build_per_category() -> list[ChatExtractionScenario]:
    return [
        ChatExtractionScenario(
            id="ce-id-01",
            name="Identity - name extraction",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-id-01-x1",
                    user_message="Hi there! My name is Jai, excited to get started.",
                    assistant_response="Welcome, Jai! Great to have you here. What are you working on right now?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-id-01-f1",
                    fact_text_pattern=r"(?i)name.*\bJai\b",
                    gold_fact_text="Student's name is Jai.",
                    category="identity",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-id-02",
            name="Identity - age and occupation",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-id-02-x1",
                    user_message="A bit about me -- I'm a 28-year-old software engineer who plays piano as a hobby.",
                    assistant_response="That's wonderful! It's great to have a creative outlet alongside your tech work. How long have you been playing?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-id-02-f1",
                    fact_text_pattern=r"(?i)(28.*(year|yo)|age.*28)",
                    gold_fact_text="Student is 28 years old.",
                    category="identity",
                    operation="add",
                ),
                ExpectedChatFact(
                    id="ce-id-02-f2",
                    fact_text_pattern=r"(?i)software engineer",
                    gold_fact_text="Student works as a software engineer.",
                    category="identity",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-bg-01",
            name="Background - years of experience",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-bg-01-x1",
                    user_message="I've been playing piano for about 3 years, mostly self-taught from YouTube videos.",
                    assistant_response="Three years of self-directed learning is impressive! YouTube can be a great resource. Let's see where you are and figure out how I can help you grow.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-bg-01-f1",
                    fact_text_pattern=r"(?i)(play|experience|learning).*(3|three)\s*year",
                    gold_fact_text="Student has been playing piano for about 3 years, mostly self-taught.",
                    category="background",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-bg-02",
            name="Background - past teacher",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-bg-02-x1",
                    user_message="I used to take lessons from Mrs. Chen at the conservatory, but I moved away last year.",
                    assistant_response="Mrs. Chen sounds like she gave you a solid foundation. What kinds of things did you work on with her?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-bg-02-f1",
                    fact_text_pattern=r"(?i)(previous|past|former|took).*(teacher|lesson|instruct).*Mrs\.?\s*Chen",
                    gold_fact_text="Student previously took lessons from Mrs. Chen at the conservatory.",
                    category="background",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-go-01",
            name="Goals - aspiration piece",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-go-01-x1",
                    user_message="I really want to be able to play Chopin's Ballade No. 1 someday. It's my dream piece.",
                    assistant_response="The Ballade No. 1 is a magnificent goal -- it demands everything from a pianist. Let's build a path toward it. What are you comfortable playing right now?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-go-01-f1",
                    fact_text_pattern=r"(?i)(goal|aspir|dream|want).*(Chopin|Ballade\s*(No\.?\s*)?1)",
                    gold_fact_text="Student's dream piece is Chopin's Ballade No. 1.",
                    category="goals",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-go-02",
            name="Goals - deadline with temporal expiry",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-go-02-x1",
                    user_message="I need to have the Beethoven sonata ready by April 15th for my recital.",
                    assistant_response="April 15th gives us about five weeks -- that's a solid timeline. Let's map out what sections need the most work so we can prioritize effectively.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-go-02-f1",
                    fact_text_pattern=r"(?i)(Beethoven|sonata).*(ready|prepar|deadline).*April\s*15",
                    gold_fact_text="Student needs the Beethoven sonata ready by April 15th for a recital.",
                    category="goals",
                    operation="add",
                    permanent=False,
                    invalid_at="2026-04-15",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-pr-01",
            name="Preferences - learning style",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-pr-01-x1",
                    user_message="I learn best when I can see it demonstrated first. Watching someone play a passage before I try it myself really helps me.",
                    assistant_response="That's a visual-first approach, and it's very effective for many pianists. I'll make sure to reference demonstrations and visual cues when we discuss technique.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-pr-01-f1",
                    fact_text_pattern=r"(?i)(learn|prefer).*(visual|demonstrat|watch|see.*first)",
                    gold_fact_text="Student learns best through visual demonstration before attempting passages.",
                    category="preferences",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-pr-02",
            name="Preferences - favorite composer",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-pr-02-x1",
                    user_message="I absolutely love Debussy, especially his Preludes. There's something about his use of color that just speaks to me.",
                    assistant_response="Debussy's Preludes are a treasure. His harmonic palette is unlike anyone else's. Have you tried any of them yet, or are they still on your wish list?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-pr-02-f1",
                    fact_text_pattern=r"(?i)(love|favor|prefer).*Debussy",
                    gold_fact_text="Student loves Debussy, especially the Preludes.",
                    category="preferences",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-rp-01",
            name="Repertoire - new piece",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-rp-01-x1",
                    user_message="I just started working on Bach's Invention No. 8 in F major. The counterpoint is tricky but I love it.",
                    assistant_response="Invention No. 8 is a wonderful choice -- the dialogue between the voices is really engaging. How are you approaching the hand independence so far?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-rp-01-f1",
                    fact_text_pattern=r"(?i)(work|learn|start|play).*(Bach|Invention\s*(No\.?\s*)?8)",
                    gold_fact_text="Student is working on Bach's Invention No. 8 in F major.",
                    category="repertoire",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-rp-02",
            name="Repertoire - piece with composer",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-rp-02-x1",
                    user_message="I'm playing Chopin's Nocturne Op. 9 No. 2. It's such a beautiful piece.",
                    assistant_response="One of the most beloved nocturnes! The singing melody in the right hand is key -- how is the rubato feeling for you?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-rp-02-f1",
                    fact_text_pattern=r"(?i)(play|work|learn).*(Chopin|Nocturne\s*Op\.?\s*9\s*(No\.?\s*)?2)",
                    gold_fact_text="Student is playing Chopin's Nocturne Op. 9 No. 2.",
                    category="repertoire",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-ev-01",
            name="Events - upcoming recital",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-ev-01-x1",
                    user_message="I have a recital coming up in 3 weeks, so I really need to tighten things up.",
                    assistant_response="Three weeks is enough time to make real progress if we're focused. Let's identify the spots that need the most polish and build a plan.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-ev-01-f1",
                    fact_text_pattern=r"(?i)recital.*(3\s*week|March\s*31|late\s*March|end.*March)",
                    gold_fact_text="Student has a recital in approximately 3 weeks (around March 31, 2026).",
                    category="events",
                    operation="add",
                    permanent=False,
                    invalid_at="2026-03-31",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-ev-02",
            name="Events - past milestone",
            category="per_category",
            exchanges=[
                ChatExchange(
                    id="ce-ev-02-x1",
                    user_message="I just passed my Grade 5 ABRSM exam last week! Scored a merit.",
                    assistant_response="Congratulations on your Grade 5 merit -- that's a real achievement! How did the aural and sight-reading sections feel?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-ev-02-f1",
                    fact_text_pattern=r"(?i)(pass|complet|achiev).*Grade\s*5.*ABRSM",
                    gold_fact_text="Student passed Grade 5 ABRSM exam with merit.",
                    category="events",
                    operation="add",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# B. UPDATE operations (6 scenarios)
# ---------------------------------------------------------------------------

def _build_update() -> list[ChatExtractionScenario]:
    return [
        ChatExtractionScenario(
            id="ce-up-01",
            name="Update - name correction",
            category="update",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-up-01-pre1",
                    fact_text_pattern=r"(?i)name.*\bJai\b",
                    gold_fact_text="Student's name is Jai.",
                    category="identity",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-up-01-x1",
                    user_message="Actually, everyone calls me Jay. I prefer that.",
                    assistant_response="Got it, Jay! I'll use that from now on. So, what are we working on today?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-up-01-f1",
                    fact_text_pattern=r"(?i)(name|call|prefer).*\bJay\b",
                    gold_fact_text="Student prefers to be called Jay.",
                    category="identity",
                    operation="update",
                    supersedes_fact_id="ce-up-01-pre1",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-up-02",
            name="Update - goal supersession",
            category="update",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-up-02-pre1",
                    fact_text_pattern=r"(?i)focus.*classical",
                    gold_fact_text="Student wants to focus on classical repertoire.",
                    category="goals",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-up-02-x1",
                    user_message="I've decided to switch my focus to jazz. I've been listening to a lot of Bill Evans and I want to learn jazz harmony and improvisation.",
                    assistant_response="That's a big shift, but Bill Evans is an incredible inspiration. Jazz will open up a whole new world of harmonic thinking for you. Shall we start with some basic voicings?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-up-02-f1",
                    fact_text_pattern=r"(?i)(focus|switch|interest).*jazz",
                    gold_fact_text="Student has switched focus from classical to jazz, inspired by Bill Evans.",
                    category="goals",
                    operation="update",
                    supersedes_fact_id="ce-up-02-pre1",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-up-03",
            name="Update - level progression",
            category="update",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-up-03-pre1",
                    fact_text_pattern=r"(?i)Grade\s*5",
                    gold_fact_text="Student is at Grade 5 level.",
                    category="background",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-up-03-x1",
                    user_message="Exciting news -- I just passed my Grade 8 exam! It was tough but I made it through.",
                    assistant_response="That's a huge milestone, congratulations! Grade 8 opens up so much repertoire. What pieces did you perform for the exam?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-up-03-f1",
                    fact_text_pattern=r"(?i)(pass|complet|achiev|reach).*Grade\s*8",
                    gold_fact_text="Student has passed Grade 8 exam, up from Grade 5.",
                    category="background",
                    operation="update",
                    supersedes_fact_id="ce-up-03-pre1",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-up-04",
            name="Update - repertoire change",
            category="update",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-up-04-pre1",
                    fact_text_pattern=r"(?i)(work|learn|play).*Bach.*Invention\s*(No\.?\s*)?8",
                    gold_fact_text="Student is working on Bach Invention No. 8.",
                    category="repertoire",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-up-04-x1",
                    user_message="I've finished the Bach Invention -- feels solid now. I'm starting on Mozart Sonata K545, the first movement.",
                    assistant_response="Wonderful progress on the Bach! Mozart K545 is a perfect next step. That opening Allegro has some deceptively tricky passages. How's the Alberti bass feeling so far?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-up-04-f1",
                    fact_text_pattern=r"(?i)(finish|complet|done).*Bach.*Invention",
                    gold_fact_text="Student has completed Bach Invention No. 8.",
                    category="repertoire",
                    operation="update",
                    supersedes_fact_id="ce-up-04-pre1",
                ),
                ExpectedChatFact(
                    id="ce-up-04-f2",
                    fact_text_pattern=r"(?i)(work|start|learn|play).*Mozart.*K\.?\s*545",
                    gold_fact_text="Student is starting Mozart Sonata K545, first movement.",
                    category="repertoire",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-up-05",
            name="Update - preference contradiction",
            category="update",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-up-05-pre1",
                    fact_text_pattern=r"(?i)practice.*morning",
                    gold_fact_text="Student prefers to practice in the morning.",
                    category="preferences",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-up-05-x1",
                    user_message="I've switched to evening practice, mornings don't work with my new schedule. I'm usually at the piano around 8 PM now.",
                    assistant_response="Evening practice can work really well -- you're often more mentally relaxed. Just watch out for fatigue in the hands if you've had a long day.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-up-05-f1",
                    fact_text_pattern=r"(?i)(switch|prefer|practice).*evening",
                    gold_fact_text="Student has switched to evening practice, usually around 8 PM.",
                    category="preferences",
                    operation="update",
                    supersedes_fact_id="ce-up-05-pre1",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-up-06",
            name="Update - multiple corrections in one exchange",
            category="update",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-up-06-pre1",
                    fact_text_pattern=r"(?i)name.*\bMike\b",
                    gold_fact_text="Student's name is Mike.",
                    category="identity",
                    operation="add",
                ),
                ExpectedChatFact(
                    id="ce-up-06-pre2",
                    fact_text_pattern=r"(?i)(level|grade).*beginner",
                    gold_fact_text="Student is at beginner level.",
                    category="background",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-up-06-x1",
                    user_message="Quick corrections -- my name is actually Michael, not Mike. And I wouldn't call myself a beginner anymore, I'd say intermediate. I've been playing for 5 years now.",
                    assistant_response="Noted, Michael! And intermediate after five years makes perfect sense. Let's make sure we're working on material that matches where you are now.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-up-06-f1",
                    fact_text_pattern=r"(?i)name.*\bMichael\b",
                    gold_fact_text="Student's name is Michael, not Mike.",
                    category="identity",
                    operation="update",
                    supersedes_fact_id="ce-up-06-pre1",
                ),
                ExpectedChatFact(
                    id="ce-up-06-f2",
                    fact_text_pattern=r"(?i)(level|self[- ]descri).*intermediate",
                    gold_fact_text="Student self-identifies as intermediate level after 5 years of playing.",
                    category="background",
                    operation="update",
                    supersedes_fact_id="ce-up-06-pre2",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# C. Temporal expiry (4 scenarios)
# ---------------------------------------------------------------------------

def _build_temporal() -> list[ChatExtractionScenario]:
    return [
        ChatExtractionScenario(
            id="ce-te-01",
            name="Temporal - recital in 3 weeks",
            category="temporal",
            exchanges=[
                ChatExchange(
                    id="ce-te-01-x1",
                    user_message="I have a recital in 3 weeks at my local music school. I need to nail the Chopin.",
                    assistant_response="Three weeks is a good window. Let's focus on the trickiest passages first and build up to full run-throughs in the final week.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-te-01-f1",
                    fact_text_pattern=r"(?i)recital.*(3\s*week|March\s*31|end.*March)",
                    gold_fact_text="Student has a recital at local music school in approximately 3 weeks (around March 31, 2026).",
                    category="events",
                    operation="add",
                    permanent=False,
                    invalid_at="2026-03-31",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-te-02",
            name="Temporal - teacher away for rest of month",
            category="temporal",
            exchanges=[
                ChatExchange(
                    id="ce-te-02-x1",
                    user_message="My regular teacher is away for the rest of March, so I'm on my own for a bit. That's why I'm using this app more.",
                    assistant_response="I'm happy to fill in while your teacher is away! We can keep your momentum going. What did your teacher have you working on before they left?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-te-02-f1",
                    fact_text_pattern=r"(?i)(teacher|instructor).*(away|absent|unavailable).*March",
                    gold_fact_text="Student's regular teacher is away for the rest of March 2026.",
                    category="events",
                    operation="add",
                    permanent=False,
                    invalid_at="2026-03-31",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-te-03",
            name="Temporal - expired fact should be absent",
            category="temporal",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-te-03-pre1",
                    fact_text_pattern=r"(?i)competition.*February",
                    gold_fact_text="Student has a competition in late February 2026.",
                    category="events",
                    operation="add",
                    permanent=False,
                    invalid_at="2026-03-05",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-te-03-x1",
                    user_message="What should I focus on this week?",
                    assistant_response="Let's do a check-in on your current pieces. Play through whatever you've been working on and we'll go from there.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[],  # The expired fact should NOT appear
        ),
        ChatExtractionScenario(
            id="ce-te-04",
            name="Temporal - permanent fact persists",
            category="temporal",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-te-04-pre1",
                    fact_text_pattern=r"(?i)name.*\bEmma\b",
                    gold_fact_text="Student's name is Emma.",
                    category="identity",
                    operation="add",
                    permanent=True,
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-te-04-x1",
                    user_message="I'm back after a long break. Feels good to play again.",
                    assistant_response="Welcome back! Taking breaks is perfectly fine -- your muscle memory will come back faster than you think. Let's ease into it.",
                    session_date="2026-09-15",
                ),
            ],
            # No new extractions expected -- the pre-existing fact persists in
            # accumulated_facts without needing to be re-extracted.
            expected_facts=[],
        ),
    ]


# ---------------------------------------------------------------------------
# D. Selectivity / adversarial (4 scenarios)
# ---------------------------------------------------------------------------

def _build_selectivity() -> list[ChatExtractionScenario]:
    return [
        ChatExtractionScenario(
            id="ce-se-01",
            name="Selectivity - empty extraction (pleasantries only)",
            category="selectivity",
            exchanges=[
                ChatExchange(
                    id="ce-se-01-x1",
                    user_message="Thanks for the tip! I'll try that.",
                    assistant_response="You're welcome, let me know how it goes!",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[],
        ),
        ChatExtractionScenario(
            id="ce-se-02",
            name="Selectivity - hypothetical, not commitment",
            category="selectivity",
            exchanges=[
                ChatExchange(
                    id="ce-se-02-x1",
                    user_message="If I were to start learning Chopin, what should I begin with? I'm just curious, not committing to anything yet.",
                    assistant_response="Great question! For a first Chopin piece, the Waltz in A minor (B.150) or Prelude in E minor (Op. 28 No. 4) are both approachable. They'll introduce you to his style without overwhelming you.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[],
        ),
        ChatExtractionScenario(
            id="ce-se-03",
            name="Selectivity - teacher info, not student info",
            category="selectivity",
            exchanges=[
                ChatExchange(
                    id="ce-se-03-x1",
                    user_message="Where did you study music?",
                    assistant_response="I draw on a wide range of teaching traditions. My approach is influenced by the Russian school of piano pedagogy, with an emphasis on tone production and musical phrasing. But enough about me -- how is your practice going this week?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[],
        ),
        ChatExtractionScenario(
            id="ce-se-04",
            name="Selectivity - buried fact in noisy exchange",
            category="selectivity",
            exchanges=[
                ChatExchange(
                    id="ce-se-04-x1",
                    user_message=(
                        "I've been thinking a lot about technique lately. Like, should I be using more "
                        "arm weight vs finger strength? I watched this video about the Taubman approach "
                        "and it was really interesting but also kind of confusing. The whole idea of "
                        "forearm rotation makes sense in theory but I'm not sure I'm doing it right. "
                        "Oh, by the way, I practice on a Yamaha C3 grand -- not sure if that matters "
                        "for technique. Anyway, do you think I should try the Taubman method or stick "
                        "with what I've been doing?"
                    ),
                    assistant_response=(
                        "The Taubman approach has excellent principles, especially around minimizing "
                        "tension. On your Yamaha C3, you'll really feel the difference when you let "
                        "the arm weight do the work rather than pressing into the keys. Let's try a "
                        "simple exercise: play a five-note scale slowly and focus on dropping into "
                        "each key from the wrist rather than pushing down with the fingers."
                    ),
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-se-04-f1",
                    fact_text_pattern=r"(?i)(practice|play|instrument|piano).*Yamaha\s*C3",
                    gold_fact_text="Student practices on a Yamaha C3 grand piano.",
                    category="background",
                    operation="add",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# E. General-purpose extraction (8 scenarios)
# ---------------------------------------------------------------------------

def _build_general() -> list[ChatExtractionScenario]:
    return [
        ChatExtractionScenario(
            id="ce-rel-01",
            name="Relationships - friend with details",
            category="general",
            exchanges=[
                ChatExchange(
                    id="ce-rel-01-x1",
                    user_message="My friend Sarah just got back from Bali, she's a painter. She brought me this beautiful watercolor of rice terraces.",
                    assistant_response="That sounds lovely! Having artistic friends can be so inspiring. Are you finding any connections between visual art and your piano playing?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-rel-01-f1",
                    fact_text_pattern=r"(?i)(friend|knows).*Sarah.*(paint|artist|Bali)",
                    gold_fact_text="Student's friend Sarah is a painter who recently returned from Bali.",
                    category="relationships",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-rel-02",
            name="Relationships - family member",
            category="general",
            exchanges=[
                ChatExchange(
                    id="ce-rel-02-x1",
                    user_message="My sister also plays piano, she's 12 and just started Grade 2. We sometimes practice together on weekends.",
                    assistant_response="That's wonderful that you share music with your sister! Playing together can be really motivating for both of you.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-rel-02-f1",
                    fact_text_pattern=r"(?i)sister.*(12|twelve|piano|Grade\s*2)",
                    gold_fact_text="Student's sister is 12 years old and plays piano, currently at Grade 2.",
                    category="relationships",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-act-01",
            name="Activities - non-piano hobby",
            category="general",
            exchanges=[
                ChatExchange(
                    id="ce-act-01-x1",
                    user_message="I've been running a lot lately, training for a half marathon in April. It's hard to find time for both running and piano practice.",
                    assistant_response="Balancing multiple passions is always tricky! Maybe shorter, focused practice sessions could help you maintain progress on both fronts.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-act-01-f1",
                    fact_text_pattern=r"(?i)(run|train|marathon).*(half\s*marathon|April)",
                    gold_fact_text="Student is training for a half marathon in April.",
                    category="activities",
                    operation="add",
                    permanent=False,
                    invalid_at="2026-04-30",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-op-01",
            name="Opinions - musical preference",
            category="general",
            exchanges=[
                ChatExchange(
                    id="ce-op-01-x1",
                    user_message="I think Debussy is overrated honestly, I much prefer Ravel. His precision and orchestration are just on another level.",
                    assistant_response="That's an interesting perspective! Ravel's craftsmanship is indeed remarkable. His piano works have a clarity that's very rewarding to play.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-op-01-f1",
                    fact_text_pattern=r"(?i)(prefer|like|favor).*Ravel|Debussy.*(overrat|less)",
                    gold_fact_text="Student prefers Ravel over Debussy, considers Debussy overrated.",
                    category="opinions",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-ctx-01",
            name="Context - living situation",
            category="general",
            exchanges=[
                ChatExchange(
                    id="ce-ctx-01-x1",
                    user_message="I just moved to Portland, still setting up my practice room. The neighbors seem cool with the piano noise so far at least!",
                    assistant_response="Congrats on the move! Having understanding neighbors is a huge plus for a pianist. What kind of setup are you going for in the practice room?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-ctx-01-f1",
                    fact_text_pattern=r"(?i)(mov|relocat|live).*Portland",
                    gold_fact_text="Student recently moved to Portland, setting up a practice room.",
                    category="context",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-ms-01",
            name="Multi-subject - another person's piano journey",
            category="general",
            exchanges=[
                ChatExchange(
                    id="ce-ms-01-x1",
                    user_message="My daughter Emma is also learning piano, she's working on her Grade 3 exam next month. I'm helping her practice scales every evening.",
                    assistant_response="That's great that you're supporting Emma's piano journey! Practicing together can make it more enjoyable for both of you.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-ms-01-f1",
                    fact_text_pattern=r"(?i)daughter.*Emma.*(Grade\s*3|piano|exam)",
                    gold_fact_text="Student's daughter Emma is learning piano and preparing for Grade 3 exam next month.",
                    category="relationships",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-se-05",
            name="Selectivity - general pleasantries zero extraction",
            category="selectivity",
            exchanges=[
                ChatExchange(
                    id="ce-se-05-x1",
                    user_message="Sure, sounds good! See you next week.",
                    assistant_response="See you then! Keep up the great practice.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[],
        ),
        ChatExtractionScenario(
            id="ce-e2e-05",
            name="E2E - mixed domain and general conversation",
            category="e2e",
            exchanges=[
                ChatExchange(
                    id="ce-e2e-05-x1",
                    user_message=(
                        "The Chopin Ballade is coming along nicely, I can play the exposition without stopping now. "
                        "Oh also, my best friend Marco is getting married next month -- I might play something at the reception!"
                    ),
                    assistant_response=(
                        "That's great progress on the exposition! And how exciting about Marco's wedding. "
                        "Playing at the reception would be a wonderful gift. Do you have a piece in mind?"
                    ),
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-e2e-05-f1",
                    fact_text_pattern=r"(?i)(Chopin|Ballade).*(exposition|progress|play.*without)",
                    gold_fact_text="Student can now play the exposition of Chopin Ballade without stopping.",
                    category="repertoire",
                    operation="add",
                ),
                ExpectedChatFact(
                    id="ce-e2e-05-f2",
                    fact_text_pattern=r"(?i)(friend|Marco).*(marry|wedding)",
                    gold_fact_text="Student's best friend Marco is getting married next month; student may play at the reception.",
                    category="relationships",
                    operation="add",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# F. End-to-end (4 scenarios)
# ---------------------------------------------------------------------------

def _build_e2e() -> list[ChatExtractionScenario]:
    return [
        ChatExtractionScenario(
            id="ce-e2e-01",
            name="E2E - extract identity then verify in context",
            category="e2e",
            exchanges=[
                ChatExchange(
                    id="ce-e2e-01-x1",
                    user_message="I'm Sarah, nice to meet you! I just downloaded this app.",
                    assistant_response="Welcome, Sarah! I'm excited to work with you. Tell me a bit about your piano background -- how long have you been playing, and what are you working on?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-e2e-01-f1",
                    fact_text_pattern=r"(?i)name.*\bSarah\b",
                    gold_fact_text="Student's name is Sarah.",
                    category="identity",
                    operation="add",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-e2e-02",
            name="E2E - multi-turn incremental with update",
            category="e2e",
            exchanges=[
                ChatExchange(
                    id="ce-e2e-02-x1",
                    user_message="My name is Alex. I'm pretty new to piano, just a couple months in.",
                    assistant_response="Welcome, Alex! A couple months is a great start. What made you decide to pick up the piano?",
                    session_date="2026-03-10",
                ),
                ChatExchange(
                    id="ce-e2e-02-x2",
                    user_message="I want to prepare for a local piano competition. I think it would motivate me to practice more seriously.",
                    assistant_response="Competitions are a fantastic motivator! Do you know when it is? That'll help us plan your preparation.",
                    session_date="2026-03-10",
                ),
                ChatExchange(
                    id="ce-e2e-02-x3",
                    user_message="Actually the competition is in June, June 15th specifically.",
                    assistant_response="June 15th -- that gives us about three months. We can build a solid preparation plan with that timeline. Let's pick your competition piece first.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-e2e-02-f1",
                    fact_text_pattern=r"(?i)name.*\bAlex\b",
                    gold_fact_text="Student's name is Alex.",
                    category="identity",
                    operation="add",
                ),
                ExpectedChatFact(
                    id="ce-e2e-02-f2",
                    fact_text_pattern=r"(?i)(new|beginner|couple\s*month)",
                    gold_fact_text="Student is a beginner, a couple months into learning piano.",
                    category="background",
                    operation="add",
                ),
                ExpectedChatFact(
                    id="ce-e2e-02-f3",
                    fact_text_pattern=r"(?i)competition.*June\s*15",
                    gold_fact_text="Student is preparing for a local piano competition on June 15th.",
                    category="goals",
                    operation="update",
                    permanent=False,
                    invalid_at="2026-06-15",
                ),
            ],
        ),
        ChatExtractionScenario(
            id="ce-e2e-03",
            name="E2E - deduplication across turns",
            category="e2e",
            exchanges=[
                ChatExchange(
                    id="ce-e2e-03-x1",
                    user_message="I'm working on a Chopin Nocturne right now, the Op. 9 No. 2.",
                    assistant_response="Lovely choice! That Nocturne has such a singing melody. How is the ornamentation going?",
                    session_date="2026-03-10",
                ),
                ChatExchange(
                    id="ce-e2e-03-x2",
                    user_message="As I mentioned, I'm practicing the Chopin Nocturne. The trills in the middle section are giving me trouble.",
                    assistant_response="Those trills can be tricky. Try practicing them slowly with a dotted rhythm first -- it helps build evenness. Make sure your hand stays relaxed.",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-e2e-03-f1",
                    fact_text_pattern=r"(?i)(work|play|practic).*Chopin.*Nocturne.*Op\.?\s*9.*No\.?\s*2",
                    gold_fact_text="Student is working on Chopin Nocturne Op. 9 No. 2.",
                    category="repertoire",
                    operation="add",
                ),
                # Only 1 fact expected, not 2 -- deduplication
            ],
        ),
        ChatExtractionScenario(
            id="ce-e2e-04",
            name="E2E - cross-source pre-existing plus student-reported",
            category="e2e",
            pre_existing_facts=[
                ExpectedChatFact(
                    id="ce-e2e-04-pre1",
                    fact_text_pattern=r"(?i)dynamics.*improve",
                    gold_fact_text="Student's dynamics have been improving over recent sessions.",
                    category="background",
                    operation="add",
                ),
            ],
            exchanges=[
                ChatExchange(
                    id="ce-e2e-04-x1",
                    user_message="I feel like my sight-reading has gotten a lot better too. I've been doing 15 minutes of sight-reading every day.",
                    assistant_response="Daily sight-reading is one of the best habits you can build! Fifteen minutes a day adds up fast. Are you using any particular books or resources?",
                    session_date="2026-03-10",
                ),
            ],
            expected_facts=[
                ExpectedChatFact(
                    id="ce-e2e-04-f1",
                    fact_text_pattern=r"(?i)sight[- ]?reading.*(improv|better|daily|15\s*min)",
                    gold_fact_text="Student practices sight-reading 15 minutes daily and feels it has improved.",
                    category="background",
                    operation="add",
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_all_scenarios() -> list[ChatExtractionScenario]:
    scenarios: list[ChatExtractionScenario] = []
    scenarios.extend(_build_per_category())
    scenarios.extend(_build_update())
    scenarios.extend(_build_temporal())
    scenarios.extend(_build_selectivity())
    scenarios.extend(_build_general())
    scenarios.extend(_build_e2e())
    return scenarios


def main() -> None:
    scenarios = build_all_scenarios()
    output_path = Path(__file__).parents[1] / "data" / "chat_scenarios.jsonl"
    save_chat_scenarios(scenarios, output_path)
    print(f"Saved {len(scenarios)} chat extraction scenarios to {output_path}")


if __name__ == "__main__":
    main()
