import streamlit as st
import PyPDF2
import os
import io
import tiktoken
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from streamlit_mic_recorder import mic_recorder

# --- TOKEN BUDGET (leave headroom under 12,000 TPM limit) ---
TOKEN_LIMIT_RUBRIC    = 2_500
TOKEN_LIMIT_PORTFOLIO = 3_500
TOKEN_LIMIT_HISTORY   = 2_000

_enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(_enc.encode(text))

def trim_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens]) + "\n... [truncated to fit token limit]"

def trim_messages_to_token_limit(messages: list, max_tokens: int) -> list:
    if not messages:
        return messages
    while len(messages) > 2:
        total = sum(count_tokens(m.content) for m in messages)
        if total <= max_tokens:
            break
        messages = messages[1:]
    return messages

# --- 1. CONFIGURATION ---

JUDGE_PERSONALITIES = {
    "Upfront": {
        "temperature": 0.2,
        "prompt": (
            "JUDGE TYPE: Upfront — critical, hole-poking, skeptical.\n"
            "Your entire purpose is to find weaknesses in the student's work. "
            "You are not mean, but you are blunt and hard to satisfy. "
            "Do not compliment anything unless it's genuinely impressive. "
            "Every question should probe a gap, inconsistency, or unsupported claim in their project. "
            "If they give a vague answer, push back. If something seems underdeveloped, say so plainly. "
            "You keep your tone direct and matter-of-fact — no warmth, no filler, no encouragement. "
            "Unlike other judges, you are NOT here to guide or explore — you are here to stress-test their work."
        ),
    },
    "Laid-back": {
        "temperature": 0.9,
        "prompt": (
            "JUDGE TYPE: Laid-back — relaxed, open-ended, student-led.\n"
            "You let the student drive the conversation. You ask broad, open questions and leave lots of room "
            "for them to go wherever they want. You never drill down or push back — if they give a short answer, "
            "you just ask another easy question. You're genuinely curious but low-pressure. "
            "Your tone is casual, like chatting with someone at a science fair. "
            "Unlike other judges, you do NOT focus on gaps or technical depth — you just want to hear them talk about their work."
        ),
    },
    "Technical": {
        "temperature": 0.3,
        "prompt": (
            "JUDGE TYPE: Technical — principle-focused, process-oriented, precise.\n"
            "You care deeply about the science, engineering, or technical logic behind the project. "
            "You want to know WHY things work, what principles were applied, what tradeoffs were made, "
            "and whether the student actually understands the underlying concepts — not just what they built. "
            "You ask things like 'What principle governs that?' or 'How did you account for X variable?' "
            "You are respectful but exacting. Vague or surface-level answers do not satisfy you. "
            "Unlike other judges, you do NOT care about visuals, presentation, or the build process — only the technical substance."
        ),
    },
    "Artistic": {
        "temperature": 0.7,
        "prompt": (
            "JUDGE TYPE: Artistic — visually focused, design-minded, aesthetic-driven.\n"
            "You evaluate everything through the lens of visual communication and design. "
            "You care about layout, color, typography, diagrams, and whether the portfolio looks intentional and polished. "
            "You ask things like 'Why did you choose that color scheme?' or 'How does your layout guide the reader's eye?' "
            "You have an eye for detail and notice when things feel cluttered, inconsistent, or visually weak. "
            "You are expressive and engaged when something looks great. "
            "Unlike other judges, you do NOT focus on technical accuracy or the build process — only the visual and design choices."
        ),
    },
    "Hands-on": {
        "temperature": 0.5,
        "prompt": (
            "JUDGE TYPE: Hands-on — process-focused, construction-minded, practical.\n"
            "You want to know exactly how things were physically made or implemented. "
            "You ask about materials, tools, fabrication steps, prototyping challenges, and real-world constraints. "
            "You're interested in what went wrong, how it was fixed, and what the student learned by actually building it. "
            "You ask things like 'What did the first prototype look like?' or 'How did you actually put that together?' "
            "You are practical, grounded, and skeptical of anything that sounds purely theoretical. "
            "Unlike other judges, you do NOT focus on visual design or abstract principles — only the tangible making process."
        ),
    },
}

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text if text.strip() else "Could not extract text from PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

def build_feedback_context(feedback_log: list[dict]) -> str:
    if not feedback_log:
        return ""
    lines = ["\n\nFEEDBACK FROM STUDENT ON YOUR PREVIOUS QUESTIONS (use this to improve):"]
    for entry in feedback_log:
        rating_str = "+1 Good question" if entry["rating"] == "up" else "-1 Poor question"
        comment_str = f" — Student comment: \"{entry['comment']}\"" if entry.get("comment") else ""
        lines.append(f"- Q: \"{entry['question'][:120]}...\" → {rating_str}{comment_str}")
    lines.append(
        "Adjust your questioning style based on the above: ask more questions like the +1 ones "
        "and avoid patterns from the -1 ones. If a comment is provided, prioritize that guidance."
    )
    return "\n".join(lines)

def get_llm(selected_judge: str, api_key: str) -> ChatGroq:
    os.environ["GROQ_API_KEY"] = api_key
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=JUDGE_PERSONALITIES[selected_judge]["temperature"],
    )

def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Send audio bytes to Groq Whisper and return transcribed text."""
    client = Groq(api_key=api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3-turbo",
        file=audio_file,
        response_format="text",
    )
    return transcription.strip()


def build_system_prompt(selected_judge: str, event_label: str, safe_rubric: str, safe_portfolio: str, feedback_context: str = "", is_opening: bool = False, theme: str = "", focused_topic: str = "") -> str:
    personality = JUDGE_PERSONALITIES[selected_judge]["prompt"]

    if is_opening:
        instructions = """
Instructions:
1. Introduce yourself in one sentence — just your name and that you're their judge. Stay in character.
2. On a new line, ask your first question. It must reflect your judge type above. Max two sentences.
3. Do NOT reference page numbers, section names, or quote the portfolio directly.
4. Sound like a real person talking, not a formal evaluator writing a report."""
        followup = ""
    else:
        instructions = ""
        if focused_topic:
            followup = f"\nRespond conversationally in 1-2 sentences, then ask exactly one follow-up question specifically about this topic: {focused_topic}. The question must stay within your judge type. Max two sentences. Never reference page numbers or quote the portfolio directly."
        else:
            followup = "\nRespond conversationally in 1-2 sentences, then ask exactly one follow-up question (max two sentences). Stay true to your judge type. Never reference page numbers or quote the portfolio directly."

    theme_section = f"\nCOMPETITION THEME:\n{theme}\nAll questions must be scoped to this theme — probe how the student\'s work addresses or connects to it.\n" if theme else ""

    return f"""{personality}

You are interviewing a student for the TSA event: {event_label}.
{instructions}
{theme_section}
EVENT RUBRIC:
{safe_rubric}

STUDENT PORTFOLIO:
{safe_portfolio}
{feedback_context}{followup}"""


def build_final_feedback_prompt(selected_judge: str, event_label: str, safe_rubric: str, safe_portfolio: str, conversation_transcript: str, theme: str = "") -> str:
    personality = JUDGE_PERSONALITIES[selected_judge]["prompt"]
    theme_section = f"\nCOMPETITION THEME:\n{theme}\n" if theme else ""

    return f"""{personality}

You just finished interviewing a student for the TSA event: {event_label}.
{theme_section}
EVENT RUBRIC:
{safe_rubric}

STUDENT PORTFOLIO:
{safe_portfolio}

FULL INTERVIEW TRANSCRIPT:
{conversation_transcript}

---

Now provide thorough, honest post-interview feedback to the student about how they performed during this interview.

Your feedback must:
- Focus entirely on the QUALITY and SUBSTANCE of the student's answers — their depth of understanding, confidence, how well they backed up claims, how completely they addressed rubric criteria, whether they demonstrated real ownership of their work, and how effectively they communicated their ideas.
- Identify 2-3 specific strengths in how they responded, with examples from the conversation.
- Identify 2-3 specific areas for improvement in how they responded, with concrete suggestions for what better answers would have looked like.
- Stay consistent with your judge personality — an Upfront judge gives blunt, critical feedback; a Laid-back judge keeps it casual; a Technical judge focuses on conceptual depth; an Artistic judge evaluates how well they communicated design intent; a Hands-on judge evaluates how concretely they described the making process.
- Do NOT comment on grammar, spelling, sentence structure, or writing mechanics in any way.
- Do NOT give a numerical score.
- Write in a direct, conversational tone — like end-of-interview verbal feedback from a real judge.

Structure your feedback with these three sections:
**What you did well**
**Where you can improve**
**One thing to work on before your next interview**"""


def generate_final_feedback(selected_judge: str, event_label: str, safe_rubric: str, safe_portfolio: str, messages: list, api_key: str, theme: str = "") -> str:
    # Build a readable transcript from the message history
    transcript_lines = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            transcript_lines.append(f"JUDGE: {msg.content}")
        elif isinstance(msg, HumanMessage):
            transcript_lines.append(f"STUDENT: {msg.content}")
    transcript = "\n\n".join(transcript_lines)

    system_prompt = build_final_feedback_prompt(
        selected_judge, event_label, safe_rubric, safe_portfolio, transcript, theme=theme
    )

    os.environ["GROQ_API_KEY"] = api_key
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.4,
    )
    response = llm.invoke([("system", system_prompt), ("human", "Please provide my post-interview feedback now.")])
    return response.content


# --- 2. FRONTEND SETUP & SIDEBAR ---

st.set_page_config(page_title="TSA Interview Prep", page_icon="🎤", layout="wide")

#  Instructions dialog 
@st.dialog("📖 How to Use TSA Interview Prep", width="large")
def show_instructions():
    st.markdown("""
## Welcome to TSA Interview Prep! 

**The goal:** To be a preparatory tool for TSA event interviews, helping you truly understand your own project and articulate responses about it.

> **A note on realism:** This AI judge is a practice tool, not a perfect simulation. Real judges will NOT phrase questions exactly like this, nor follow the same flow, and may focus on completely different topics. This tool is designed to quiz you thoroughly and make sure you can explain every part of your project clearly.

---

### Step 1 — Get Your Free Groq API Key

This app uses a free AI service called **Groq**. You need a free account to use it — no credit card required.

1. Click the button below to open Groq's website
2. Click **"Sign Up"** (top right), you can sign up with Google for speed
3. Once logged in, click your name or icon in the top-right → **"API Keys"**
4. Click **"Create API Key"**, give it any name, then click **"Submit"**
5. **Copy the key** (starts with `gsk_...`) and paste it into the sidebar
""")
    st.link_button("🔗 Get Free Groq API Key →", "https://console.groq.com/keys", use_container_width=True, type="primary")
    st.markdown("""
---

### Step 2 — Fill In the Sidebar

| Field | Required? | What to put |
|---|---|---|
| **Groq API Key** | Yes | The key you just copied |
| **Event Rubric** | Yes | Paste or upload your TSA event rubric PDF |
| **Event Name** | Optional | e.g. "Biotechnology Design" |
| **Competition Theme** | Optional | Any theme your competition uses, short or long |
| **Judge Personality** | Yes | Pick the style of judge you want to practice with |
| **Project Info** | Yes | Upload your portfolio PDF, or type a description if you don't have one |

---

### ‍ Step 3 — Choose Your Judge

Pick a judge personality that matches how you want to practice:

- **Upfront** — Blunt and critical. Will poke holes in your work. Great for stress-testing.
- **Laid-back** — Relaxed, open-ended questions. Good for getting comfortable talking about your project.
- **Technical** — Focuses on the science and engineering logic behind your work.
- **Artistic** — Evaluates visual design, layout, and how you communicate aesthetics.
- **Hands-on** — Wants to know exactly how you built it, step by step.

---

###  Step 4 — Run Your Interview

Once everything is filled in, the judge will introduce themselves and ask an opening question.

- **Type your answer** in the chat box and press Enter
- After each answer, the judge responds and asks a follow-up
- Use the **Next Question Focus** bar to direct the judge toward a specific topic you want to practice
- Hit **"Random question"** if you want the judge to go wherever they want
- Rate each question +1 or -1. The judge adapts its style based on your ratings

---

### Step 5 — Get Feedback

When you're done practicing, click **"✅ Finish & Get Feedback"** in the sidebar.

The judge gives you a full breakdown of:
- What you did well in your responses
- Where your answers were weak or incomplete
- One concrete thing to work on before the real interview

---

### Reset

Use **"🔄 Reset Interview"** in the sidebar to start over with a new judge, rubric, or portfolio at any time.
""")
    if st.button("Got it, let's go! 🚀", use_container_width=True, type="primary"):
        st.session_state["instructions_seen"] = True
        st.rerun()

#  Page title 
st.title("🎤 TSA Interview Prep")
st.caption("Practice your TSA event interview with an AI judge tailored to your portfolio.")

with st.sidebar:
    #  Finish button — always at top of sidebar so it's always on screen 
    if st.session_state.get("interview_started") and not st.session_state.get("show_final_feedback"):
        if st.button("Finish & Get Feedback", use_container_width=True, type="primary", key="finish_sidebar"):
            st.session_state["show_final_feedback"] = True
            st.rerun()
        st.markdown("---")

    #  Instructions button
    if st.button("How to use this app", use_container_width=True):
        st.session_state["open_instructions"] = True
        st.rerun()

    st.header("⚙️ Interview Settings")

    st.subheader("Groq API Key")
    st.caption("Free, no credit card needed. Click below to get yours in ~1 minute.")
    st.link_button("🔗 Get Free API Key →", "https://console.groq.com/keys", use_container_width=True)
    raw_key = st.text_input("Paste your API key here", type="password", placeholder="gsk_...", key="api_key_input")
    if raw_key.strip():
        st.session_state["groq_api_key"] = raw_key.strip()
    api_key = st.session_state.get("groq_api_key", "")
    if api_key:
        st.success("API key saved", icon="🔑")

    st.markdown("---")

    st.subheader("📋 Event Rubric")
    rubric_input_method = st.radio(
        "How would you like to provide your rubric?",
        ["Paste as text", "Upload as PDF"],
        key="rubric_input_method",
        horizontal=True,
    )

    rubric_text = ""
    if rubric_input_method == "Paste as text":
        rubric_text = st.text_area(
            "Paste your event rubric here",
            height=200,
            placeholder="Paste your full TSA event rubric, scoring criteria, or event guidelines here...",
            key="rubric_text_input",
        ).strip()
    else:
        uploaded_rubric = st.file_uploader(
            "Upload Rubric PDF", type=["pdf"], key="rubric_pdf_upload"
        )
        if uploaded_rubric:
            with st.spinner("Reading rubric PDF..."):
                rubric_text = extract_text_from_pdf(uploaded_rubric)
            if rubric_text and not rubric_text.startswith("Error") and not rubric_text.startswith("Could not"):
                st.success("✅ Rubric loaded from PDF.")
            else:
                st.error(f"I'm sorry, we ran into a bug! Bug Description: {rubric_text or 'Failed to extract text from the PDF.'}")
                rubric_text = ""

    event_name_input = st.text_input(
        "Event name (optional)",
        placeholder="e.g. Biotechnology Design, Software Development...",
        key="event_name_input",
    ).strip()

    theme_input = st.text_area(
        "Competition theme (optional)",
        height=100,
        placeholder="e.g. 'Sustainability' or 'Design solutions that address accessibility challenges in urban environments...'",
        key="theme_input",
    ).strip()

    st.markdown("---")

    judge_labels = {
        "Upfront":    "⚡ Upfront — critical & hole-poking",
        "Laid-back":  "😎 Laid-back — open-ended & student-led",
        "Technical":  "🔬 Technical — principles & process",
        "Artistic":   "🎨 Artistic — visuals & aesthetic",
        "Hands-on":   "🔧 Hands-on — how it was built",
    }
    selected_judge = st.selectbox(
        "Judge Personality",
        list(JUDGE_PERSONALITIES.keys()),
        format_func=lambda k: judge_labels[k],
        key="judge_select",
    )

    st.markdown("---")
    st.subheader("📁 Project Information")
    portfolio_input_method = st.radio(
        "How will you provide your project?",
        ["Upload Portfolio PDF", "Describe your project"],
        key="portfolio_input_method",
        horizontal=True,
    )

    uploaded_portfolio = None
    project_description = ""
    if portfolio_input_method == "Upload Portfolio PDF":
        uploaded_portfolio = st.file_uploader("Upload Portfolio (PDF)", type=["pdf"], key="pdf_upload")
    else:
        project_description = st.text_area(
            "Project description",
            height=220,
            placeholder="Describe your project in as much detail as you can: what it is, how it works, what you built or designed, key decisions you made, results or findings, etc.",
            key="project_description_input",
        ).strip()

    st.markdown("---")
    st.subheader("Feedback Summary")
    feedback_log = st.session_state.get("feedback_log", [])
    if feedback_log:
        thumbs_up = sum(1 for f in feedback_log if f["rating"] == "up")
        thumbs_down = len(feedback_log) - thumbs_up
        st.write(f"👍 {thumbs_up}  |  👎 {thumbs_down}  |  Total rated: {len(feedback_log)}")
    else:
        st.write("No feedback submitted yet.")

    st.markdown("---")

    if st.button("Reset Interview", use_container_width=True):
        for key in ["messages", "interview_started", "portfolio_text", "feedback_log",
                    "pending_feedback_idx", "show_feedback_form", "groq_api_key",
                    "show_final_feedback", "final_feedback_text"]:
            st.session_state.pop(key, None)
        st.rerun()

# --- 3. STATE MANAGEMENT ---

for key, default in [
    ("messages", []),
    ("interview_started", False),
    ("portfolio_text", ""),
    ("feedback_log", []),
    ("pending_feedback_idx", None),
    ("show_feedback_form", False),
    ("show_final_feedback", False),
    ("final_feedback_text", None),
    ("instructions_seen", False),
    ("open_instructions", False),
    ("last_audio_len", -1),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- 3b. Dialog trigger (single call per run prevents double-open error) ---
if not st.session_state["instructions_seen"] or st.session_state["open_instructions"]:
    st.session_state["instructions_seen"] = True   # mark seen immediately so reruns don't retrigger
    st.session_state["open_instructions"] = False
    show_instructions()

# --- 4. MAIN CHAT AREA ---

if not api_key:
    st.warning("⬅️ Enter your free Groq API key in the sidebar to begin.")
    st.stop()

if not rubric_text:
    st.info("⬅️ Provide your TSA event rubric in the sidebar (paste text or upload PDF).")
    st.stop()

# Require either a PDF upload or a project description
has_project_info = bool(uploaded_portfolio) or bool(project_description)
if not has_project_info:
    if portfolio_input_method == "Upload Portfolio PDF":
        st.info("⬅️ Upload your TSA portfolio PDF in the sidebar to start the interview.")
    else:
        st.info("⬅️ Enter a description of your project in the sidebar to start the interview.")
    st.stop()

event_label = event_name_input if event_name_input else "the selected TSA event"
theme_label = theme_input if theme_input else ""

# --- 4a. Initialize interview ---
if not st.session_state.interview_started:
    spinner_msg = "Analyzing your portfolio and rubric..." if uploaded_portfolio else "Setting up your interview..."
    with st.spinner(spinner_msg):
        if uploaded_portfolio:
            raw_portfolio = extract_text_from_pdf(uploaded_portfolio)
        else:
            raw_portfolio = "PROJECT DESCRIPTION (provided verbally by student, no PDF portfolio):\n" + project_description

        safe_rubric    = trim_to_token_limit(rubric_text, TOKEN_LIMIT_RUBRIC)
        safe_portfolio = trim_to_token_limit(raw_portfolio, TOKEN_LIMIT_PORTFOLIO)
        st.session_state.portfolio_text = safe_portfolio

        if count_tokens(rubric_text) > TOKEN_LIMIT_RUBRIC:
            st.warning(f"Rubric was trimmed to ~{TOKEN_LIMIT_RUBRIC} tokens to fit the model limit.")
        if uploaded_portfolio and count_tokens(raw_portfolio) > TOKEN_LIMIT_PORTFOLIO:
            st.warning(f"Portfolio was trimmed to ~{TOKEN_LIMIT_PORTFOLIO} tokens to fit the model limit.")

        st.session_state.interview_started = True

        system_instruction = build_system_prompt(
            selected_judge, event_label, safe_rubric, safe_portfolio, is_opening=True, theme=theme_label
        )

        llm = get_llm(selected_judge, api_key)
        try:
            initial_response = llm.invoke([("system", system_instruction)])
            st.session_state.messages.append(AIMessage(content=initial_response.content))
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                st.session_state.interview_started = False
                st.error(
                        "⏱️ **You've hit the Groq API rate limit.** "
                        "This happens when you've made too many free requests in a short period. "
                        "You can continue by: waiting until tomorrow for your limit to reset, "
                        "using a different Groq account, or upgrading your current account at "
                        "[console.groq.com](https://console.groq.com/keys)."
                    )
            else:
                st.error(f"I'm sorry, we ran into a bug! Bug Description: {e}")
            st.stop()

# --- 4b. Render chat messages ---
for i, msg in enumerate(st.session_state.messages):
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)

        if role == "assistant" and i > 0:
            already_rated = any(f["msg_idx"] == i for f in st.session_state.feedback_log)
            if already_rated:
                existing = next(f for f in st.session_state.feedback_log if f["msg_idx"] == i)
                icon = "+1" if existing["rating"] == "up" else "-1"
                icon = "👍" if existing["rating"] == "up" else "👎"
                st.caption(f"{icon} Feedback recorded.")
            else:
                col1, col2, col3 = st.columns([1, 1, 8])
                with col1:
                    if st.button("👍", key=f"up_{i}", help="Good question"):
                        st.session_state.pending_feedback_idx = i
                        st.session_state.pending_feedback_rating = "up"
                        st.session_state.show_feedback_form = True
                        st.rerun()
                with col2:
                    if st.button("👎", key=f"down_{i}", help="Poor question"):
                        st.session_state.pending_feedback_idx = i
                        st.session_state.pending_feedback_rating = "down"
                        st.session_state.show_feedback_form = True
                        st.rerun()

# --- 4c. Feedback form ---
if st.session_state.show_feedback_form and st.session_state.pending_feedback_idx is not None:
    idx = st.session_state.pending_feedback_idx
    rating = st.session_state.pending_feedback_rating
    rated_question = st.session_state.messages[idx].content

    with st.form(key="feedback_form", clear_on_submit=True):
        icon = "👍" if rating == "up" else "👎"
        st.markdown(f"**{icon} Optional: Add a comment about this question**")
        comment = st.text_input(
            "What made it good or bad? (press Enter or Submit to skip)",
            placeholder="e.g. 'Too vague' or 'Great follow-up on the rubric gap'",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Submit Feedback")
        if submitted:
            st.session_state.feedback_log.append({
                "msg_idx": idx,
                "question": rated_question,
                "rating": rating,
                "comment": comment.strip(),
            })
            st.session_state.show_feedback_form = False
            st.session_state.pending_feedback_idx = None
            st.rerun()

# --- 4d. Final interview feedback panel ---
if st.session_state.get("show_final_feedback"):
    st.divider()
    st.subheader("📝 Post-Interview Feedback")

    # Generate feedback only once and cache it
    if not st.session_state.get("final_feedback_text"):
        human_turns = [m for m in st.session_state.messages if isinstance(m, HumanMessage)]
        if not human_turns:
            st.warning("You haven't answered any questions yet! Have a conversation with the judge first.")
        else:
            with st.spinner("Generating your feedback report..."):
                safe_rubric    = trim_to_token_limit(rubric_text, TOKEN_LIMIT_RUBRIC)
                safe_portfolio = trim_to_token_limit(st.session_state.portfolio_text, TOKEN_LIMIT_PORTFOLIO)
                try:
                    feedback_text = generate_final_feedback(
                        selected_judge, event_label, safe_rubric, safe_portfolio,
                        st.session_state.messages, api_key, theme=theme_label
                    )
                    st.session_state["final_feedback_text"] = feedback_text
                except Exception as e:
                    if "rate" in str(e).lower() or "429" in str(e):
                        st.error(
                            "⏱️ **You've hit the Groq API rate limit.** "
                            "This happens when you've made too many free requests in a short period. "
                            "You can continue by: waiting until tomorrow for your limit to reset, "
                            "using a different Groq account, or upgrading your current account at "
                            "[console.groq.com](https://console.groq.com/keys)."
                        )
                    else:
                        st.error(f"I'm sorry, we ran into a bug! Bug Description: {e}")

    if st.session_state.get("final_feedback_text"):
        st.markdown(st.session_state["final_feedback_text"])

    st.stop()

# --- 4e. Chat input & response (hidden once feedback is shown) ---

def run_judge_response(user_input: str, focused_topic: str = ""):
    """Append user message, call the judge, append AI response, rerun."""
    st.session_state.messages.append(HumanMessage(content=user_input))

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("The judge is thinking..."):
            feedback_context = build_feedback_context(st.session_state.feedback_log)

            safe_rubric    = trim_to_token_limit(rubric_text, TOKEN_LIMIT_RUBRIC)
            safe_portfolio = trim_to_token_limit(st.session_state.portfolio_text, TOKEN_LIMIT_PORTFOLIO)

            trimmed_messages = trim_messages_to_token_limit(
                list(st.session_state.messages), TOKEN_LIMIT_HISTORY
            )

            system_instruction = build_system_prompt(
                selected_judge, event_label, safe_rubric, safe_portfolio,
                feedback_context=feedback_context, is_opening=False,
                theme=theme_label, focused_topic=focused_topic
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_instruction),
                MessagesPlaceholder(variable_name="messages"),
            ])

            llm = get_llm(selected_judge, api_key)
            chain = prompt | llm
            try:
                response = chain.invoke({"messages": trimmed_messages})
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    st.error(
                        "⏱️ **You've hit the Groq API rate limit.** "
                        "This happens when you've made too many free requests in a short period. "
                        "You can continue by: waiting until tomorrow for your limit to reset, "
                        "using a different Groq account, or upgrading your current account at "
                        "[console.groq.com](https://console.groq.com/keys)."
                    )
                else:
                    st.error(f"I'm sorry, we ran into a bug! Bug Description: {e}")
                return

    st.rerun()


#  Answer input: typed or spoken 
user_input = st.chat_input("Type your answer to the judge...")

# Speech-to-text row
with st.container():
    st.caption("🎙️ **Prefer to speak?** Click the mic, give your answer out loud, then click stop. Your words will be transcribed and sent automatically.")
    audio = mic_recorder(
        start_prompt="🎙️ Start recording",
        stop_prompt="⏹️ Stop & send",
        just_once=True,
        use_container_width=True,
        key="mic_recorder",
    )

# Transcribe and submit spoken answer if new audio came in
if audio and audio.get("bytes"):
    audio_bytes = audio["bytes"]
    # Only process if this is genuinely new audio (compare byte length as a cheap fingerprint)
    last_len = st.session_state.get("last_audio_len", -1)
    if len(audio_bytes) != last_len:
        st.session_state["last_audio_len"] = len(audio_bytes)
        with st.spinner("Transcribing your answer..."):
            try:
                transcribed = transcribe_audio(audio_bytes, api_key)
                if transcribed:
                    run_judge_response(transcribed)
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    st.error(
                        "⏱️ **You've hit the Groq API rate limit.** "
                        "You can continue by: waiting until tomorrow for your limit to reset, "
                        "using a different Groq account, or upgrading your current account at "
                        "[console.groq.com](https://console.groq.com/keys)."
                    )
                else:
                    st.error(f"I'm sorry, we ran into a bug! Bug Description: {e}")

# Topic control row — sits just below the mic
with st.container():
    st.caption("**Next question focus** — type a topic to direct the judge's next question, or leave blank for a random one.")
    topic_col, btn_col_topic, btn_col_random = st.columns([5, 2, 2])
    with topic_col:
        next_topic = st.text_input(
            "Next question topic",
            placeholder="e.g. your testing process, cost analysis, design tradeoffs...",
            label_visibility="collapsed",
            key="next_topic_input",
        ).strip()
    with btn_col_topic:
        ask_topic_btn = st.button(
            "Ask on this topic",
            use_container_width=True,
            disabled=not next_topic,
            key="ask_topic_btn",
        )
    with btn_col_random:
        ask_random_btn = st.button(
            "Random question",
            use_container_width=True,
            key="ask_random_btn",
        )

# Handle regular answer submission via chat input
if user_input:
    run_judge_response(user_input)

# Handle topic-focused next question (no answer text needed — judge pivots on their own)
if ask_topic_btn and next_topic:
    run_judge_response(
        f"[Student has asked to be questioned on: {next_topic}]",
        focused_topic=next_topic,
    )

# Handle random next question
if ask_random_btn:
    run_judge_response(
        "[Student has asked for a random follow-up question.]",
        focused_topic="",
    )