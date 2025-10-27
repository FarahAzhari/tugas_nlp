import nltk
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import stanza

# Download resources (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Indonesian stopwords from NLTK
from nltk.corpus import stopwords
indonesian_stopwords = set(stopwords.words('indonesian'))

# Stanza pipeline for Indonesian (POS + Dependency Parsing)
stanza.download('id')
nlp = stanza.Pipeline('id')

# Create stemmer for Indonesian
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Example sentence
sentence = "Saya suka membaca buku di perpustakaan."

print("Original Sentence:", sentence)

# 1. Tokenization
tokens = nltk.word_tokenize(sentence)
print("\nTokens:", tokens)

# 2. Normalization (Lowercasing)
tokens = [t.lower() for t in tokens]
print("\nLowercased:", tokens)

# 3. Remove stopwords
tokens = [t for t in tokens if t not in indonesian_stopwords]
print("\nStopword Removal:", tokens)

# 4. Remove punctuation
tokens = [t for t in tokens if t not in string.punctuation]
print("\nPunctuation Removal:", tokens)

# 5. Stemming
stems = [stemmer.stem(t) for t in tokens]
print("\nStemming:", stems)

# Join back into processed sentence for NLP parsing
processed_sentence = " ".join(stems)

# 6 & 7. POS Tagging + Dependency Parsing with Stanza
doc = nlp(processed_sentence)

print("\nPOS Tagging & Dependency Parsing:")
for sent in doc.sentences:
    for word in sent.words:
        print(f"{word.text}\t{word.upos}\tHead={word.head}\tDepRel={word.deprel}")
