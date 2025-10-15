from collections import defaultdict

def inverted_search(query, documents, make_snippet):
    """Perform simple inverted index search."""
    if not documents:
        return []

    # Build inverted index
    inverted_index = defaultdict(set)
    for fname, doc in documents.items():
        for word in doc["processed"].split():
            inverted_index[word].add(fname)

    # Process query
    query_terms = query.lower().split()
    matched_files = set()
    for term in query_terms:
        matched_files |= inverted_index.get(term, set())

    results = []
    for fname in matched_files:
        doc = documents[fname]
        results.append({
            "filename": fname,
            "title": doc["title"],
            "snippet": make_snippet(doc["content"], query),
            "similarity": None
        })

    return results
