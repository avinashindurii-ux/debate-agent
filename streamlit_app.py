"""Streamlit front end for the Debate Agent. Uses the GROQ_API_KEY from your own terminal."""
import html
import os

import streamlit as st

st.set_page_config(page_title="Debate Agent", page_icon="🎙️", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@600;800&family=Newsreader:ital,wght@0,400;1,400&display=swap');
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;height:0}
.block-container {padding-top:2.2rem;max-width:780px}
.hero {text-align:center;margin-bottom:1.6rem}
.hero h1 {font:800 clamp(2.4rem,8vw,4rem)/1 'Bricolage Grotesque',sans-serif;letter-spacing:-.03em;margin:0 0 .6rem}
.hero .for {color:#5BB0FF}.hero .against {color:#FF6F8E}
.hero p {font:400 1.15rem/1.5 'Newsreader',serif;color:#A9A6BC;max-width:46ch;margin:0 auto}
.round {text-align:center;font:600 .9rem 'Bricolage Grotesque',sans-serif;color:#8C89A3;margin:1.6rem 0 .6rem;letter-spacing:.04em}
.bubble {max-width:84%;padding:14px 18px;margin:10px 0;border-radius:18px;font:400 1.15rem/1.55 'Newsreader',serif}
.bubble b {display:block;font:600 .85rem 'Bricolage Grotesque',sans-serif;margin-bottom:4px}
.bubble.for {background:#16304F;border:1px solid #2C5E96;border-bottom-left-radius:4px;margin-right:auto}
.bubble.for b {color:#7CC0FF}
.bubble.against {background:#4A1B2C;border:1px solid #96344F;border-bottom-right-radius:4px;margin-left:auto;text-align:right}
.bubble.against b {color:#FF8DA6}
.mod {margin:14px auto;max-width:90%;padding:10px 16px;border-left:3px solid #F5C451;background:#1E1D17;border-radius:4px;font:400 1rem/1.5 'Newsreader',serif;color:#CFCBB8}
.mod b {font:600 .85rem 'Bricolage Grotesque',sans-serif;color:#F5C451}
.mod .flag {color:#F5C451;display:block;margin-top:4px;font-size:.92rem}
.recap {margin-top:1.8rem;padding:20px 24px;border-radius:14px;background:linear-gradient(135deg,#16304F,#4A1B2C);border:1px solid #3A3550}
.recap h3 {font:800 1.4rem 'Bricolage Grotesque',sans-serif;margin:0 0 8px}
.recap p {font:400 1.2rem/1.6 'Newsreader',serif;margin:0}
.thinking {text-align:center;color:#8C89A3;font:italic 1.05rem 'Newsreader',serif;margin:14px 0}
.thinking i {display:inline-block;width:6px;height:6px;border-radius:50%;background:#8C89A3;margin:0 2px;animation:p 1.2s infinite}
.thinking i:nth-child(2){animation-delay:.15s}.thinking i:nth-child(3){animation-delay:.3s}
@keyframes p{0%,60%,100%{opacity:.2}30%{opacity:1}}
@media (prefers-reduced-motion:reduce){.thinking i{animation:none}}
div.stButton > button {border-radius:999px}
@media (max-width:560px){.bubble{max-width:96%}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hero"><h1><span class="for">For</span> vs <span class="against">Against</span></h1>'
    "<p>Pick a topic. Two AI agents take opposite sides, and a moderator calls out anyone "
    "who dodges, repeats, or makes things up.</p></div>",
    unsafe_allow_html=True,
)

if not os.environ.get("GROQ_API_KEY"):
    st.error("No Groq API key found. Stop the app, set GROQ_API_KEY in your terminal, "
             "then run `streamlit run streamlit_app.py` again.")
    st.code('set GROQ_API_KEY=your-key-here        (Windows cmd)\n'
            '$env:GROQ_API_KEY="your-key-here"     (PowerShell)\n'
            'export GROQ_API_KEY="your-key-here"   (Mac/Linux)')
    st.stop()

import debate  # imported after the key check, since it creates the Groq client on import

IDEAS = ["Should remote work be the default?",
         "Should universities drop entrance exams?",
         "Is nuclear power the answer to climate change?"]


def pick(idea):
    st.session_state["topic"] = idea


topic = st.text_input("What should they argue about?", key="topic",
                      placeholder="Type a topic, or pick one below")
cols = st.columns(len(IDEAS))
for col, idea in zip(cols, IDEAS):
    col.button(idea, on_click=pick, args=(idea,), use_container_width=True)
rounds = st.slider("Rounds", 2, 5, 3)
start = st.button("Start the debate", type="primary", use_container_width=True)


def esc(text):
    return html.escape(text).replace("$", "&#36;")


if start:
    if len(topic.split()) < 3:
        st.warning("Enter a topic of at least three words.")
        st.stop()

    status = st.empty()
    total = rounds

    def thinking(label):
        status.markdown(f'<div class="thinking">{label} <i></i><i></i><i></i></div>',
                        unsafe_allow_html=True)

    def show(markup):
        st.markdown(markup, unsafe_allow_html=True)

    def on_event(kind, d):
        status.empty()
        if kind == "start":
            thinking("Setting the stage")
        elif kind == "round":
            show(f'<div class="round">ROUND {d["round"]}</div>')
            thinking("For is thinking")
        elif kind == "turn":
            side = "for" if d["side"] == "A" else "against"
            show(f'<div class="bubble {side}"><b>{side.capitalize()}</b>{esc(d["text"])}</div>')
            thinking("Against is thinking" if side == "for" else "The moderator is reviewing")
        elif kind == "moderator":
            flags = ", ".join(f.replace("A:", "For:").replace("B:", "Against:") for f in d["flags"])
            extra = f'<span class="flag">{esc(flags)}</span>' if flags else ""
            if d["unsupported"].lower() != "none":
                extra += f'<span class="flag">Unsupported claim: {esc(d["unsupported"])}</span>'
            show(f'<div class="mod"><b>Moderator</b><br>{esc(d["summary"])}{extra}</div>')
            thinking("Writing the recap" if d["round"] >= total else "Next round loading")
        elif kind == "redirect":
            show('<div class="mod"><b>Moderator</b><br>Both sides have been circling the same '
                 "point. Back to the core question.</div>")
        elif kind == "recap":
            show(f'<div class="recap"><h3>The short version</h3><p>{esc(d["text"])}</p></div>')

    try:
        debate.run_debate(topic.strip(), rounds, on_event=on_event)
    except Exception as e:
        st.error(f"The debate stopped early: {e}")
    status.empty()
