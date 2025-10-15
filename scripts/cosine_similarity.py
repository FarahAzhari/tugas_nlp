from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.01

def cosine_search(query, documents, make_snippet):
    """Perform cosine similarity search using TF-IDF."""
    if not documents:
        return []

    vectorizer = TfidfVectorizer()
    doc_texts = [doc["processed"] for doc in documents.values()]
    doc_vectors = vectorizer.fit_transform(doc_texts)
    query_vec = vectorizer.transform([query])

    similarity = cosine_similarity(query_vec, doc_vectors).flatten()

    results = []
    for (fname, doc), sim in zip(documents.items(), similarity):
        if sim >= SIMILARITY_THRESHOLD:
            results.append({
                "filename": fname,
                "title": doc["title"],
                "snippet": make_snippet(doc["content"], query),
                "similarity": round(float(sim), 3)
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results
