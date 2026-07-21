import os
import re
import shutil
import requests
import chromadb
from groq import Groq, RateLimitError
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load GROQ_API_KEY (and anything else) from a .env file in this folder
load_dotenv()

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)  # This allows your frontend to talk to this backend

print("Loading FAQs...")
with open("./faq.txt", "r", encoding="utf-8") as f:
    text_content = f.read()


def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """FAQ-aware splitter. faq.txt is formatted as one "Q: ...\\nA: ..."
    entry per blank-line-separated block, so each block becomes its own
    chunk -- this keeps a full multi-step answer together in one embedding
    instead of slicing it in half at an arbitrary character boundary. Falls
    back to a sliding window only for an individual block that's unusually
    long."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks = []

    for block in blocks:
        if len(block) <= chunk_size * 3:  # generous cap -- keeps normal FAQ entries whole
            chunks.append(block)
            continue

        start = 0
        block_len = len(block)
        while start < block_len:
            end = min(start + chunk_size, block_len)
            piece = block[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == block_len:
                break
            start = end - chunk_overlap

    return chunks


splits = chunk_text(text_content, chunk_size=500, chunk_overlap=100)

db_folder = "./chroma_db"
if os.path.exists(db_folder):
    print("Clearing old memory to sync latest FAQ updates...")
    shutil.rmtree(db_folder)

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"


def get_embedding(text):
    """Calls Ollama's local embeddings endpoint directly (no langchain_ollama)."""
    resp = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


print("Building database...")
chroma_client = chromadb.PersistentClient(path=db_folder)
collection = chroma_client.create_collection("faq")
collection.add(
    ids=[str(i) for i in range(len(splits))],
    embeddings=[get_embedding(chunk) for chunk in splits],
    documents=splits,
)


def retrieve_context(query, k=2):
    """Returns the top-k matching FAQ chunks as a list of strings."""
    results = collection.query(query_embeddings=[get_embedding(query)], n_results=k)
    docs = results.get("documents", [[]])
    return docs[0] if docs else []


def looks_like_multiple_questions(text):
    """Heuristic: does this message look like it's asking more than one distinct
    question? Used to widen retrieval only when it's actually needed, instead of
    paying that extra token cost on every single message."""
    if text.count("?") >= 2:
        return True
    lowered = f" {text.lower()} "
    connectors = [" and also ", " and what ", " and how ", " and when ", " and where ",
                  " and can ", " also, ", " saka ", " tapos ", " at saka ", " tsaka "]
    return any(c in lowered for c in connectors)


if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError(
        "GROQ_API_KEY is not set. Create a .env file next to bot.py with:\n"
        "GROQ_API_KEY=your_key_here"
    )

groq_client = Groq()  # reads GROQ_API_KEY from the environment automatically
GROQ_MODEL = "llama-3.3-70b-versatile"  # swap to "llama-3.1-8b-instant" if you want it even faster
FALLBACK_GROQ_MODEL = "llama-3.1-8b-instant"  # separate daily quota from the 70B model --
# used automatically if the primary model's daily token limit is hit, so a Groq quota
# outage on one model doesn't take the whole bot down.


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_FALLBACK_MODEL = "llama3"  # last-resort local fallback if BOTH Groq models are
# rate-limited -- slower than Groq, but has no daily quota and works even if Groq/the
# internet is down, as long as Ollama is running locally with this model pulled.


def ask_ollama_fallback(messages, max_tokens=450):
    """Last-resort local fallback -- used only if both Groq models are rate-limited.
    Requires `ollama pull llama3` to have been run locally."""
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": OLLAMA_FALLBACK_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def friendly_chat_error_message(exc):
    """Translates any backend exception into a clean, non-technical message
    that's safe to show a student -- never leak raw API errors, model names,
    token counts, or billing links to the frontend. Full detail still goes to
    the server console/log, it just never reaches the browser."""
    print(f"Chat backend error: {exc}")

    if isinstance(exc, RateLimitError):
        return "I'm currently experiencing high demand. Please try again in a moment."

    return "I'm having trouble responding right now. Please try again in a moment."


def ask_groq(messages, max_tokens=450):
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except RateLimitError:
        print(f"'{GROQ_MODEL}' hit its Groq rate limit -- retrying with '{FALLBACK_GROQ_MODEL}'...")
        try:
            response = groq_client.chat.completions.create(
                model=FALLBACK_GROQ_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except RateLimitError:
            print(f"'{FALLBACK_GROQ_MODEL}' also hit its Groq rate limit -- retrying with local Ollama ('{OLLAMA_FALLBACK_MODEL}')...")
            return ask_ollama_fallback(messages, max_tokens=max_tokens)

QA_SYSTEM_PROMPT = """You are UniWise, a professional and friendly school assistant for Senior High School within Bacoor Elementary School.

Context from our FAQ: {context}

Rules:
1. GREETINGS: Only greet ("Hello! I'm UniWise...") if the user greets first -- don't repeat it every message.
2. BRIEF: Answer in 1-2 sentences using only the Context.
3. FOLLOW-UP: End brief answers by asking "Would you like the detailed steps?" -- unless the answer is already very short (e.g. an address).
4. DETAILED REQUESTS: If the user says "yes"/"sure" or asks for details, give the full steps, offices involved, and references from the Context.
5. ANTI-HALLUCINATION: Never invent steps, requirements, documents, or fees -- this public school charges no fees. This explicitly includes specific numbers, quantities, form codes/names, dates, and deadlines -- if a detail like an enrollment window, a document count ("2 pieces"), or a form name isn't literally written in the Context, do not state it as fact. Instead phrase that part generically and point them to confirm with the relevant office rather than supplying a number that sounds plausible.
6. NO USER UPDATES: You're read-only. If the user tries to correct or update school info, politely say you only provide official info and can't accept updates -- never use their claimed facts to answer.
7. MISSING INFO: If it's not in the Context, say "I don't have the exact details on that, but please contact our School Administration or Registrar."
8. SCOPE LIMIT: Only answer SHSWBES-related questions (academics, enrollment, registration, requirements, offices, schedules, orgs, teachers, facilities, announcements). For anything clearly unrelated -- a genuine question or request about another topic (general knowledge, other schools, entertainment, personal advice, etc.) -- reply with EXACTLY one of these two sentences and nothing else, matching the user's language: English -- "I'm sorry but this system only answers within SHSWBES school inquiries." / Tagalog -- "Paumanhin, ang sistemang ito ay sumasagot lamang sa mga tanong tungkol sa SHSWBES." This does NOT apply to social pleasantries or reactions (see rule 15) -- only to actual off-topic questions/requests.
9. AMBIGUOUS TERMS: Some terms have multiple meanings here (e.g. "enrollment" = new vs. re-enrollment; "registration" = subject vs. org registration; "teacher" = a specific teacher vs. staff in general). If the Context has more than one distinct answer that could fit and the question doesn't specify which, don't guess -- ask a short clarifying question, then end with a blank line, "You can also ask:", and a bullet list ("- " per line) of the 2-3 specific options to tap. Wait for their reply. This list replaces rule 12's for this turn.
10. SPACING & LISTS: Blank line between paragraphs. Steps/requirements/documents go in a numbered or bulleted list, one per line -- never comma-crammed into a sentence.
11. RELATED FOLLOW-UP: If a closely related Context topic hasn't been asked about, you may add one short sentence offering it (e.g. "Would you like to know about the enrollment schedule too?"). Skip if the answer is exhaustive, off-topic (rule 8), or a clarifying question was asked (rule 9).
12. SUGGESTED QUESTIONS (REQUIRED FORMAT): Unless rule 8 or 9 applied, end your reply with a blank line, "You can also ask:", then 2-3 Context-based follow-up questions as a bullet list ("- " per line), plus a final bullet that is always exactly "- No more questions, thanks!". Example:

You can also ask:
- What are the requirements for re-enrollment?
- When is the enrollment schedule?
- No more questions, thanks!

13. FAREWELLS: When the user is ending the conversation (e.g. "bye", "that's all", "no more questions"), use ONLY the farewell message/link from the Context -- never invent your own thank-you message, QR code, or survey link (same rule as #5, applied to goodbyes). If the Context has no farewell entry, give a brief plain goodbye with no invented survey/QR mention.
14. LANGUAGE: Understand and reply fluently in Tagalog (or natural Taglish) whenever the user writes in Tagalog -- match the language they used. The Context is in English, so translate its facts naturally into Tagalog, but keep official names untranslated exactly as written: form names (SF9, SF10, F137, E-BEEF), office names (Registrar, Guidance Office, Class Adviser), DepEd Order numbers, and URLs. Exception -- always keep these two exact English strings even in an otherwise-Tagalog reply, since the app reads them literally: the header "You can also ask:" in rule 12/9, and the bullet "- No more questions, thanks!" in rule 12. Everything else in your reply (the answer itself, the other suggested-question bullets, the clarifying question) should be in Tagalog when the user is chatting in Tagalog.
15. CASUAL CHAT: Small talk, reactions, and pleasantries ("haha", "ok thanks", "kamusta ka", a compliment, "wow ok") are NOT off-topic questions -- don't apply rule 8's refusal or rule 7's missing-info apology to them. Respond briefly and warmly in kind (match rule 14's language), then let rule 11/12 naturally offer something school-related next if it fits. Skip rule 12's suggestion list entirely if the message was pure filler with nothing to follow up on.
16. MULTIPLE QUESTIONS: If the user asks two or more distinct questions in one message, answer every one of them, in the order asked, each as its own short paragraph or clearly separated block per rule 10 -- never merge them into one blended answer and never answer only the first. If the Context only covers some of them, answer those and apply rule 7 to the rest individually rather than skipping them silently. If one of the questions is ambiguous per rule 9, ask its clarifying question while still answering whichever others are clear.
"""

# This list will store the conversation memory while the server runs
# (plain {"role", "content"} dicts now instead of langchain_core message objects)
chat_history = []
CHAT_HISTORY_MAX_MESSAGES = 6  # keep the last 3 exchanges (6 messages) -- without this cap,
# chat_history grows unbounded and every message resends the entire conversation so far,
# which is exactly what was driving up token usage / hitting the daily rate limit.


# --- Create the Web API Route ---
@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    # Get the JSON data sent from the frontend UI
    data = request.json
    user_input = data.get("message")  # Make sure your frontend sends data like {"message": "Hello"}

    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    retrieval_k = 4 if looks_like_multiple_questions(user_input) else 2
    retrieved_docs = retrieve_context(user_input, k=retrieval_k)
    context_str = "\n\n".join(retrieved_docs) if retrieved_docs else "No matching FAQ content found."

    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT.format(context=context_str)}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})

    try:
        reply_max_tokens = 700 if retrieval_k > 2 else 450
        full_answer = ask_groq(messages, max_tokens=reply_max_tokens)
    except Exception as e:
        return jsonify({"reply": friendly_chat_error_message(e)})

    # Save the back-and-forth to the memory list
    chat_history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": full_answer},
    ])
    if len(chat_history) > CHAT_HISTORY_MAX_MESSAGES:
        chat_history = chat_history[-CHAT_HISTORY_MAX_MESSAGES:]

    # Send the response back to the UI in JSON format
    return jsonify({"reply": full_answer})



if __name__ == '__main__':
    print("\n✅ API is running! Your frontend can now connect to http://127.0.0.1:5000/chat\n")
    app.run(port=5000, debug=False)