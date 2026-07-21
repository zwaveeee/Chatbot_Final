import json
import os
import re
import uuid
from datetime import datetime
from flask import Blueprint, jsonify, request

faq_bp = Blueprint("faq_bp", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FAQ_DATA_FILE = os.path.join(DATA_DIR, "faq_data.json")


def ensure_data_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(FAQ_DATA_FILE):
        with open(FAQ_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "questions": []
            }, f, indent=2)


def load_faq_data():
    ensure_data_file()
    try:
        with open(FAQ_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"questions": []}
            if "questions" not in data or not isinstance(data["questions"], list):
                data["questions"] = []
            return data
    except Exception:
        return {"questions": []}


def save_faq_data(data):
    ensure_data_file()
    with open(FAQ_DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def now_iso():
    return datetime.utcnow().isoformat()


def normalize_question(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


# ── SIMILARITY HELPERS ────────────────────────────────────────
_STOP = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","can","shall","to","of","in","on",
    "at","for","with","from","by","about","as","into","before",
    "after","and","but","or","not","no","it","i","you","me","my",
    "your","we","they","who","what","when","where","why","how",
    "which","that","this","there","their","any","all","some",
    "please","tell","give","show","explain","get","hi","hello"
}

def _keywords(text):
    words = normalize_question(text).split()
    return [w for w in words if len(w) > 2 and w not in _STOP]

def _similarity(text_a, text_b):
    """Jaccard + containment similarity. Returns 0.0 – 1.0."""
    wa = set(_keywords(text_a))
    wb = set(_keywords(text_b))
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    if not inter:
        return 0.0
    jaccard     = inter / len(wa | wb)
    containment = inter / min(len(wa), len(wb))
    return max(jaccard, containment * 0.85)

SIMILARITY_THRESHOLD = 0.55   # tune if needed

def find_similar_approved(questions, question_text):
    """Return the approved FAQ most similar to question_text, or None."""
    best_item  = None
    best_score = 0.0
    norm_input = normalize_question(question_text)

    for item in questions:
        if item.get("status") != "approved":
            continue
        raw_q = item.get("question", "") or item.get("normalized_question", "")
        if not raw_q:
            continue
        # Exact normalized match wins immediately
        if item.get("normalized_question") == norm_input:
            return item
        score = _similarity(question_text, raw_q)
        if score > best_score and score >= SIMILARITY_THRESHOLD:
            best_score = score
            best_item  = item
    return best_item

def find_similar_pending(questions, faq_item):
    """Return all 'new' questions similar to faq_item (excluding itself)."""
    results = []
    for item in questions:
        if item is faq_item:
            continue
        if item.get("status") != "new":
            continue
        raw_q = item.get("question", "") or item.get("normalized_question", "")
        if not raw_q:
            continue
        score = _similarity(faq_item.get("question", ""), raw_q)
        if score >= SIMILARITY_THRESHOLD:
            results.append(item)
    return results


    return str(text or "").strip()


def make_question_id():
    return f"faq_{uuid.uuid4().hex[:12]}"


def find_question_by_id(questions, faq_id):
    for item in questions:
        if item.get("id") == faq_id:
            return item
    return None


def get_sorted_all_questions(questions):
    def sort_key(item):
        return item.get("updated_at") or item.get("created_at") or ""
    return sorted(questions, key=sort_key, reverse=True)


def get_top_faqs(questions, limit=10):
    approved = [q for q in questions if q.get("status") == "approved"]
    approved_sorted = sorted(
        approved,
        key=lambda x: (
            int(x.get("count", 0)),
            x.get("updated_at") or x.get("created_at") or ""
        ),
        reverse=True
    )

    top = approved_sorted[:limit]
    for index, item in enumerate(top):
        item["rank"] = index + 1
    return top


def get_new_questions(questions):
    new_items = [q for q in questions if q.get("status") == "new"]
    return sorted(
        new_items,
        key=lambda x: (
            int(x.get("count", 0)),
            x.get("updated_at") or x.get("created_at") or ""
        ),
        reverse=True
    )


def build_faq_insights_payload(questions):
    all_questions = get_sorted_all_questions(questions)
    top_faqs = get_top_faqs(questions)
    new_questions = get_new_questions(questions)

    return {
        "top_faqs": top_faqs,
        "new_questions": new_questions,
        "all_questions": all_questions
    }


@faq_bp.get("/api/faq-insights")
def get_faq_insights():
    data = load_faq_data()
    questions = data.get("questions", [])

    return jsonify({
        "success": True,
        "data": build_faq_insights_payload(questions)
    })


@faq_bp.get("/api/chatbot/faqs")
def get_chatbot_faqs():
    data = load_faq_data()
    questions = data.get("questions", [])

    approved = [q for q in questions if q.get("status") == "approved"]
    approved_sorted = sorted(
        approved,
        key=lambda x: (
            int(x.get("count", 0)),
            x.get("updated_at") or x.get("created_at") or ""
        ),
        reverse=True
    )

    return jsonify({
        "success": True,
        "data": approved_sorted
    })


@faq_bp.post("/api/chatbot/questions")
def log_chatbot_question():
    payload = request.get_json(silent=True) or {}
    question = sanitize_question(payload.get("question"))

    if not question:
        return jsonify({
            "success": False,
            "error": "Question is required."
        }), 400

    normalized = normalize_question(question)
    if not normalized:
        return jsonify({
            "success": False,
            "error": "Question is invalid."
        }), 400

    data = load_faq_data()
    questions = data.get("questions", [])

    existing = None
    for item in questions:
        if item.get("normalized_question") == normalized:
            existing = item
            break

    timestamp = now_iso()

    if existing:
        existing["count"] = int(existing.get("count", 0)) + 1
        existing["updated_at"] = timestamp

        # keep original answer/status unless admin changes it
        if not existing.get("question"):
          existing["question"] = question
    else:
        # ── Before creating a new "new" entry, check if an approved FAQ
        #    already covers this question.  If so, just increment its count
        #    — no duplicate pending entry gets created.
        approved_match = find_similar_approved(questions, question)
        if approved_match:
            approved_match["count"] = int(approved_match.get("count", 0)) + 1
            approved_match["updated_at"] = timestamp
        else:
            questions.append({
                "id": make_question_id(),
                "question": question,
                "normalized_question": normalized,
                "answer": "",
                "count": 1,
                "status": "new",
                "source": "chatbot",
                "created_at": timestamp,
                "updated_at": timestamp
            })

    save_faq_data(data)

    return jsonify({
        "success": True,
        "message": "Question logged successfully."
    })


@faq_bp.post("/api/faq-insights")
def create_faq_manually():
    payload = request.get_json(silent=True) or {}
    question = sanitize_question(payload.get("question"))
    answer = str(payload.get("answer") or "").strip()
    approve = bool(payload.get("approve", True))

    if not question:
        return jsonify({
            "success": False,
            "error": "Question is required."
        }), 400

    normalized = normalize_question(question)
    timestamp = now_iso()

    data = load_faq_data()
    questions = data.get("questions", [])

    existing = None
    for item in questions:
        if item.get("normalized_question") == normalized:
            existing = item
            break

    if existing:
        existing["question"] = question
        existing["answer"] = answer
        existing["status"] = "approved" if approve else existing.get("status", "new")
        existing["updated_at"] = timestamp
        if not existing.get("source"):
            existing["source"] = "admin"
        save_faq_data(data)

        return jsonify({
            "success": True,
            "data": existing,
            "message": "FAQ updated successfully."
        })

    new_item = {
        "id": make_question_id(),
        "question": question,
        "normalized_question": normalized,
        "answer": answer,
        "count": 0,
        "status": "approved" if approve else "new",
        "source": "admin",
        "created_at": timestamp,
        "updated_at": timestamp
    }

    questions.append(new_item)
    save_faq_data(data)

    return jsonify({
        "success": True,
        "data": new_item,
        "message": "FAQ created successfully."
    })


@faq_bp.put("/api/faq-insights/<faq_id>")
def update_faq_insight(faq_id):
    payload = request.get_json(silent=True) or {}
    question = sanitize_question(payload.get("question"))
    answer = str(payload.get("answer") or "").strip()
    approve = bool(payload.get("approve", False))

    data = load_faq_data()
    questions = data.get("questions", [])

    item = find_question_by_id(questions, faq_id)
    if not item:
        return jsonify({
            "success": False,
            "error": "FAQ not found."
        }), 404

    if question:
        item["question"] = question
        item["normalized_question"] = normalize_question(question)

    item["answer"] = answer
    if approve:
        item["status"] = "approved"

    item["updated_at"] = now_iso()

    save_faq_data(data)

    return jsonify({
        "success": True,
        "data": item,
        "message": "FAQ updated successfully."
    })


@faq_bp.post("/api/faq-insights/<faq_id>/approve")
def approve_faq_insight(faq_id):
    data = load_faq_data()
    questions = data.get("questions", [])

    item = find_question_by_id(questions, faq_id)
    if not item:
        return jsonify({
            "success": False,
            "error": "FAQ not found."
        }), 404

    item["status"] = "approved"
    item["updated_at"] = now_iso()

    # ── Auto-merge similar pending questions into this approved FAQ ──
    # Find all "new" questions similar to the one just approved,
    # add their counts to this FAQ, then remove them so they don't
    # appear as duplicates in the pending section.
    similar_pending = find_similar_pending(questions, item)
    merged_count    = 0
    merged_ids      = []

    for dup in similar_pending:
        item["count"] = int(item.get("count", 0)) + int(dup.get("count", 0))
        merged_count += int(dup.get("count", 0))
        merged_ids.append(dup.get("id"))

    if merged_ids:
        data["questions"] = [
            q for q in questions
            if q.get("id") not in merged_ids
        ]
        item["updated_at"] = now_iso()

    save_faq_data(data)

    return jsonify({
        "success": True,
        "data": item,
        "merged_count":     merged_count,
        "merged_questions": len(merged_ids),
        "message": (
            f"FAQ approved and {len(merged_ids)} similar question(s) merged "
            f"(+{merged_count} to ask count)."
            if merged_ids else "FAQ approved successfully."
        )
    })


@faq_bp.delete("/api/faq-insights/<faq_id>")
def delete_faq_insight(faq_id):
    data = load_faq_data()
    questions = data.get("questions", [])

    item = find_question_by_id(questions, faq_id)
    if not item:
        return jsonify({
            "success": False,
            "error": "FAQ not found."
        }), 404

    data["questions"] = [q for q in questions if q.get("id") != faq_id]
    save_faq_data(data)

    return jsonify({
        "success": True,
        "message": "FAQ deleted successfully."
    })