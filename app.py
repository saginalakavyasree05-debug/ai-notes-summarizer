import os
import re
import io
from flask import Flask, render_template, request, send_file, jsonify
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract

# If tesseract is not on PATH or you installed in custom location, uncomment and set:
# Example Windows path:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = None

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB uploads
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(stream):
    text = []
    try:
        with pdfplumber.open(stream) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
    except Exception as e:
        print("PDF read error:", e)
    return "\n".join(text)

def extract_text_from_docx(stream):
    text = []
    try:
        doc = Document(stream)
        for p in doc.paragraphs:
            if p.text:
                text.append(p.text)
    except Exception as e:
        print("DOCX read error:", e)
    return "\n".join(text)

def extract_text_from_image(stream):
    try:
        img = Image.open(stream).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print("Image OCR error:", e)
        return ""

def detect_subject(text):
    txt = text.lower()
    keywords = {
        "math": ["solve", "equation", "angle", "integral", "theorem", "derivative", "algebra", "geometry"],
        "science": ["experiment", "hypothesis", "cell", "atom", "photosynthesis", "gravity", "biology", "chemical"],
        "social": ["history", "economy", "revolution", "society", "culture", "government", "ancient"],
        "english": ["poem", "literature", "novel", "metaphor", "prose", "author", "writing", "grammar"]
    }
    counts = {k: sum(1 for kw in kws if kw in txt) for k, kws in keywords.items()}
    subject = max(counts, key=lambda k: counts[k]) if max(counts.values()) > 0 else "general"
    return subject

def summarize_for_students(text, max_sentences=5):
    # Basic cleaning
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "", []

    # Sentence tokenize
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return text, sentences

    # TF-IDF scoring for sentences
    try:
        vect = TfidfVectorizer(stop_words="english")
        X = vect.fit_transform(sentences)
        # score sentences by sum of tfidf weights
        scores = X.sum(axis=1).A1
    except Exception as e:
        # fallback: length-based scoring
        scores = [len(s) for s in sentences]

    # pick top sentences (preserve original order)
    ranked_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
    top_k_idx = sorted(ranked_idx[:min(max_sentences, len(sentences))])
    summary_sentences = [sentences[i] for i in top_k_idx]
    summary = " ".join(summary_sentences)

    # Extract important terms for highlighting: top TF-IDF words across chosen sentences
    try:
        # Fit vectorizer on full text to get feature names
        top_features = []
        global_vect = TfidfVectorizer(stop_words="english", max_features=200)
        global_X = global_vect.fit_transform([text])
        feature_array = global_vect.get_feature_names_out()
        # compute term importance by TF-IDF value across entire text
        tfidf_scores = global_X.toarray()[0]
        # top N terms
        top_n = 12
        term_indices = tfidf_scores.argsort()[::-1][:top_n]
        top_features = [feature_array[i] for i in term_indices if tfidf_scores[i] > 0]
    except Exception:
        top_features = []

    return summary, top_features

def underline_terms(summary, terms):
    if not summary or not terms:
        return summary
    # sort terms by length desc to avoid partial overlaps first
    terms_sorted = sorted(set(terms), key=lambda t: -len(t))
    out = summary
    # We'll insert underline tags — do it carefully to avoid messing indices: use regex replace with word boundaries
    for term in terms_sorted:
        if not term or len(term) < 2:
            continue
        pattern = re.compile(re.escape(term), flags=re.IGNORECASE)
        out = pattern.sub(lambda m: f"<u>{m.group(0)}</u>", out)
    return out

@app.route("/", methods=["GET", "POST"])
def home():
    raw_text = ""
    subject = "general"
    summary_html = ""
    extracted = ""
    error_message = None

    if request.method == "POST":
        text_input = request.form.get("text", "").strip()
        file = request.files.get("file")

        if file and file.filename != "":
            filename = file.filename
            ext = filename.rsplit(".", 1)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                error_message = f"Unsupported file type: {ext}"
            else:
                # read file bytes into BytesIO for different readers
                stream = io.BytesIO(file.read())
                stream.seek(0)
                if ext == "pdf":
                    raw_text = extract_text_from_pdf(stream)
                elif ext == "docx":
                    stream.seek(0)
                    raw_text = extract_text_from_docx(stream)
                elif ext in ("png", "jpg", "jpeg"):
                    stream.seek(0)
                    raw_text = extract_text_from_image(stream)
                else:  # txt
                    try:
                        stream.seek(0)
                        raw_text = stream.read().decode("utf-8", errors="ignore")
                    except Exception:
                        raw_text = ""
                extracted = raw_text
        else:
            raw_text = text_input
            extracted = text_input

        if not raw_text:
            error_message = error_message or "No text found. Paste text or upload a readable file (txt/pdf/docx/image)."

        if not error_message:
            subject = detect_subject(raw_text)
            # choose summary length heuristics based on input size
            sent_count = len(sent_tokenize(raw_text))
            if sent_count <= 4:
                max_sents = 1
            elif sent_count <= 10:
                max_sents = 2
            elif sent_count <= 30:
                max_sents = 3
            elif sent_count <= 70:
                max_sents = 4
            else:
                max_sents = 5

            summary, top_terms = summarize_for_students(raw_text, max_sentences=max_sents)

            # refine terms for subject: boost domain-relevant words (optional simple heuristic)
            if subject == "math":
                # prefer numeric and math-related tokens
                pass  # for now our tfidf picks decent terms

            # Underline top terms inside the summary (no boxes)
            summary_html = underline_terms(summary, top_terms[:8])  # up to 8 underlines
    return render_template(
        "ai.html",
        raw_text=extracted,
        subject=subject.title(),
        summary_html=summary_html,
        error_message=error_message
    )

@app.route("/download_summary", methods=["POST"])
def download_summary():
    data = request.form.get("summary_text", "")
    if not data:
        return jsonify({"error": "No summary provided"}), 400
    return send_file(io.BytesIO(data.encode("utf-8")), as_attachment=True, download_name="summary.txt", mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True)