from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords (run once)
nltk.download('stopwords')
stop_words = stopwords.words('indonesian') 

# Sample Indonesian sentences
sentences = [
    "Saya suka makan nasi",
    "Nasi goreng adalah makanan favorit saya",
    "Saya tidak suka makan sayur",
    "Makanan Indonesia sangat lezat",
    "Saya suka makanan pedas"
]

# Initialize the Sastrawi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocessing: stemming + lowercasing
stemmed_sentences = [stemmer.stem(sentence.lower()) for sentence in sentences]

print("Stemmed Sentences:")
for sent in stemmed_sentences:
    print(sent)
print()

# Initialize TF-IDF Vectorizer with stopword removal
vectorizer = TfidfVectorizer(stop_words=stop_words)

# Fit and transform the stemmed sentences
tfidf_matrix = vectorizer.fit_transform(stemmed_sentences)

# Get the vocabulary (feature names)
feature_names = vectorizer.get_feature_names_out()

# Display TF-IDF scores
for i, sentence in enumerate(stemmed_sentences):
    print(f"Sentence {i+1}: {sentence}")
    for word, score in zip(feature_names, tfidf_matrix[i].toarray()[0]):
        if score > 0:
            print(f"  {word}: {score:.4f}")
    print()
