import os
import re
import string
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from topic_modelling_lda import perform_lda

# Import search algorithms
from cosine_similarity import cosine_search
from inverted_index import inverted_search

# NLTK preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("punkt")
    nltk.download('punkt_tab')
    nltk.download("stopwords")
    nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FOLDER = os.path.join(BASE_DIR, "../templates/search-engines")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "../uploads/search-engines")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask app
app = Flask(__name__, template_folder=TEMPLATE_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"txt"}

# -----------------------------
# HELPERS
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_text(text: str) -> str:
    """Lowercase, remove punctuation, tokenize, remove stopwords, lemmatize."""
    if not text:
        return ""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def make_snippet(raw_text: str, query: str, length: int = 200) -> str:
    """Build a snippet around the query words."""
    if not raw_text:
        return ""

    query_words = [w for w in query.lower().strip().split() if w not in stop_words]
    raw_lower = raw_text.lower()

    first_pos = -1
    for word in query_words:
        pos = raw_lower.find(word)
        if pos != -1 and (first_pos == -1 or pos < first_pos):
            first_pos = pos

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

    for word in query_words:
        snippet = re.sub(rf"\b({re.escape(word)})\b", r"<b>\1</b>", snippet, flags=re.IGNORECASE)
    return " ".join(snippet.split())


def load_documents():
    """Load .txt documents from upload folder."""
    docs = {}
    for fname in os.listdir(UPLOAD_FOLDER):
        if fname.endswith(".txt"):
            path = os.path.join(UPLOAD_FOLDER, fname)
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    continue
                title = lines[0].strip()
                content = " ".join(line.strip() for line in lines[1:])
                docs[fname] = {
                    "title": title,
                    "content": content,
                    "processed": preprocess_text(content),
                }
    return docs


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def search():
    query = request.values.get("query", "")
    engine = request.values.get("engine", "cosine")  # default engine
    docs = load_documents()

    results = []
    if query:
        if engine == "cosine":
            results = cosine_search(query, docs, make_snippet)
        elif engine == "inverted":
            results = inverted_search(query, docs, make_snippet)

    return render_template("index.html", results=results, query=query, engine=engine)


@app.route("/doc/<filename>")
def doc_view(filename):
    docs = load_documents()
    if filename in docs:
        return render_template("doc.html", doc=docs[filename], filename=filename)
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
            return redirect(url_for("admin"))

    files = os.listdir(UPLOAD_FOLDER)
    return render_template("admin.html", files=files)


@app.route("/delete/<filename>", methods=["POST"])
def delete_file(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        os.remove(path)
    return redirect(url_for("admin"))

@app.route("/lda")
def lda_dashboard():
    docs = load_documents()
    topics, doc_topics = perform_lda(docs)
    return render_template(
        "index.html",
        mode="lda",
        topics=topics,
        doc_topics=doc_topics,
        engine=None,
        results=None,
        query=None
    )


if __name__ == "__main__":
    app.run(debug=True)
