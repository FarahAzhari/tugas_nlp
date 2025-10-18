# topic_modelling_lda.py
import os
import nltk
import string
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Reuse preprocessing
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def perform_lda(documents, num_topics=5):
    """Perform LDA topic modelling and return topics + doc-topic mapping."""
    if not documents:
        return [], []

    # Preprocess all docs
    processed_docs = [preprocess_text(doc["content"]) for doc in documents.values()]

    # Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(text) for text in processed_docs]

    # Train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha="auto"
    )

    # Extract topics
    topics = []
    for idx, topic in lda_model.show_topics(num_topics=num_topics, num_words=7, formatted=False):
        topics.append({
            "topic_id": idx,
            "words": [word for word, _ in topic]
        })

    # Determine dominant topic per document
    doc_topics = []
    for i, doc_bow in enumerate(corpus):
        topic_probs = lda_model.get_document_topics(doc_bow)
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        doc_topics.append({
            "filename": list(documents.keys())[i],
            "title": documents[list(documents.keys())[i]]["title"],
            "dominant_topic": dominant_topic
        })

    return topics, doc_topics
