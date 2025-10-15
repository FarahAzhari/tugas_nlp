import os
import re
import string
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# make sure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Base directory (scripts/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Templates folder → ../templates/search-engines
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "../templates/search-engines")

# Uploads folder → ../uploads/search-engines
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../uploads/search-engines")

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app config
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"txt"}

# In-memory document store
documents = {}  # {filename: {"title": str, "content": str, "processed": str}}
vectorizer = None
doc_vectors = None

# Similarity threshold
SIMILARITY_THRESHOLD = 0.01
RESULTS_PER_PAGE = 10


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, stopwords, lemmatize."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def make_snippet(raw_text: str, query: str, length: int = 200) -> str:
    """Make a snippet around query words, bolding only meaningful words (stopwords ignored)."""
    if not raw_text:
        return ""

    query_words = [w for w in query.lower().strip().split() if w not in stop_words]
    raw_lower = raw_text.lower()

    # Find first occurrence of any query word
    first_pos = -1
    for word in query_words:
        pos = raw_lower.find(word)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

    # Build snippet around the first matched word
    if first_pos == -1:
        snippet = raw_text[:length]
    else:
        start = max(0, first_pos - length // 3)
        end = min(len(raw_text), first_pos + (2 * length) // 3)
        snippet = raw_text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(raw_text):
            snippet = snippet + "..."

    # Highlight remaining query words
    for word in query_words:
        snippet = re.sub(
            rf"\b({re.escape(word)})\b",  # full word only
            r"<b>\1</b>",
            snippet,
            flags=re.IGNORECASE
        )

    return " ".join(snippet.split())


def load_documents():
    """Load all txt files into memory and rebuild vectorizer"""
    global documents, vectorizer, doc_vectors
    documents = {}

    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.endswith(".txt"):
            path = os.path.join(UPLOAD_FOLDER, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    continue
                title = lines[0].strip()
                content = " ".join(line.strip() for line in lines[1:])
                documents[fname] = {
                    "title": title,
                    "content": content,
                    "processed": preprocess_text(content),
                }

    if documents:
        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform([doc["processed"] for doc in documents.values()])
    else:
        vectorizer, doc_vectors = None, None


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def search():
    query = request.values.get("query", "")
    page = int(request.args.get("page", 1))
    results = []
    total_results = 0

    if query and vectorizer:
        query_vec = vectorizer.transform([preprocess_text(query)])
        similarity = cosine_similarity(query_vec, doc_vectors).flatten()
        doc_list = list(documents.items())

        # Sort all by similarity (no cutoff for pagination)
        ranked = sorted(
            [(fname, documents[fname], sim) for (fname, doc), sim in zip(doc_list, similarity)],
            key=lambda x: x[2],
            reverse=True,
        )

        # Filter out results below threshold
        filtered = [(fname, doc, sim) for fname, doc, sim in ranked if sim >= SIMILARITY_THRESHOLD]

        total_results = len(filtered)

        # Pagination logic
        start = (page - 1) * RESULTS_PER_PAGE
        end = start + RESULTS_PER_PAGE
        ranked_page = filtered[start:end]

        for fname, doc, sim in ranked_page:
            results.append(
                {
                    "filename": fname,
                    "title": (doc["title"][:60] + "...") if len(doc["title"]) > 60 else doc["title"],
                    "snippet": make_snippet(doc["content"], query),
                    "similarity": round(float(sim), 3),
                }
            )

    return render_template(
        "index.html",
        results=results,
        query=query,
        page=page,
        results_per_page=RESULTS_PER_PAGE,
        total_results=total_results,
    )


@app.route("/doc/<filename>")
def doc_view(filename):
    if filename in documents:
        return render_template("doc.html", doc=documents[filename], filename=filename)
    return "Document not found", 404


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            load_documents()
            return redirect(url_for("admin"))

    files = list(documents.keys())
    return render_template("admin.html", files=files)


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
        load_documents()
    return redirect(url_for("admin"))


# Initial load
load_documents()

if __name__ == "__main__":
    app.run(debug=True)
