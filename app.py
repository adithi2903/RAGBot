"""
The chat interface (dark theme, streaming). After `python ingest.py`, run:

    python app.py

Open the local URL it prints (e.g. http://127.0.0.1:7860).
"""
import gradio as gr
from rag import answer, answer_stream, audit

WAIT = "🔎 *Reading the Act…*"


def _extract_answer(raw):
    """Pull the text inside <answer>...</answer>; '' if the tag hasn't appeared yet."""
    if "<answer>" in raw:
        return raw.split("<answer>", 1)[1].split("</answer>")[0].strip()
    return ""


def _format(ans, hits):
    """Add the verification badge and the cited sources under the answer."""
    status, unsupported = audit(ans, hits)
    if status == "ok":
        badge = "✅ *Citations verified against the retrieved sources.*"
    elif status == "warn":
        badge = ("⚠️ *Caution: cited section(s) " + ", ".join(unsupported)
                 + " were not found in the retrieved sources — verify manually.*")
    else:
        badge = "ℹ️ *No specific section cited.*"
    sources = "\n".join(
        f"- **{h['source_file']}**"
        + (f", {h['section']}" if h.get("section") else "")
        + f" (p.{h['page']})"
        for h in hits
    )
    return f"{ans}\n\n{badge}\n\n---\n**Sources**\n{sources}"


def chat_fn(message, history):
    """Stream the answer token-by-token, then finalize with badge + sources.
    Falls back to a single non-streaming call if streaming ever fails."""
    try:
        raw, hits = "", []
        for r, h in answer_stream(message):
            raw, hits = r, h
            inner = _extract_answer(raw)
            yield inner if inner else WAIT   # hide the model's thinking until the answer opens

        # finalize
        final = _extract_answer(raw)
        if not final:
            # the model never emitted a clean answer -> robust non-streaming retry
            ans, hits = answer(message)
            final = ans
        final = final or "I could not find this in the provided documents."
        yield _format(final, hits) if hits else final
    except Exception:
        try:
            ans, hits = answer(message)
            yield _format(ans, hits) if hits else ans
        except Exception as e:
            yield f"⚠️ Sorry, something went wrong: {e}\n\nPlease try again."


# Dark grey / black theme
dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.gray,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

# Force the dark look via CSS variables (works regardless of Gradio version)
CSS = """
:root, .gradio-container, .dark {
  --body-background-fill:#0b0b0d;
  --background-fill-primary:#141417;
  --background-fill-secondary:#1b1b1f;
  --block-background-fill:#141417;
  --block-border-color:#2a2a30;
  --border-color-primary:#2a2a30;
  --body-text-color:#e8e8ea;
  --body-text-color-subdued:#9a9aa2;
  --color-accent:#6b7280;
  --color-accent-soft:#26262c;
  --button-primary-background-fill:#3a3a42;
  --button-primary-background-fill-hover:#4a4a54;
  --button-primary-text-color:#ffffff;
  --input-background-fill:#1b1b1f;
}
.gradio-container { background:#0b0b0d !important; color:#e8e8ea !important; }
footer { display:none !important; }
#title h1 { color:#f5f5f7 !important; font-weight:700; letter-spacing:-0.5px; }
#title p  { color:#9a9aa2 !important; }
/* chat bubbles */
.message.user, [data-testid="user"] {
  background:#2a2a30 !important; color:#fff !important; border-radius:14px !important; }
.message.bot, [data-testid="bot"] {
  background:#17171b !important; color:#e8e8ea !important;
  border:1px solid #2a2a30 !important; border-radius:14px !important; }
.disclaimer { color:#6a6a72 !important; font-size:0.8rem; text-align:center; }
"""

with gr.Blocks(title="Indian Tax Law Assistant") as demo:
    gr.Markdown(
        "# Indian Tax Law Assistant\n"
        "Answers grounded in official income-tax PDFs, with cited sources. "
        "Powered by NVIDIA Nemotron + RAG.",
        elem_id="title",
    )
    gr.ChatInterface(
        fn=chat_fn,
        examples=[
            "What is the deduction available under section 80C?",
            "What changed between the Income-tax Act 1961 and 2025?",
            "What is the due date for filing an income tax return?",
        ],
    )
    gr.Markdown(
        "Educational project — not professional tax advice. "
        "Verify against the cited source documents.",
        elem_classes="disclaimer",
    )

if __name__ == "__main__":
    demo.launch(theme=dark_theme, css=CSS)