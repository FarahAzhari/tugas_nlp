# word_vectors.py
import os
from gensim.models import Word2Vec
import logging

# Setup logging untuk melihat progres di konsol
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# --- Path dan Model Global ---
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "word2vec.model")

# Variabel global untuk menampung model yang sudah dimuat
# Ini jauh lebih efisien daripada memuat model di setiap request
model = None

def load_word_vector_model():
    """
    Memuat model Word2Vec yang sudah dilatih ke dalam memori.
    Dipanggil saat aplikasi Flask dimulai.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Memuat model Word2Vec dari {MODEL_PATH}...")
            model = Word2Vec.load(MODEL_PATH)
            print("Model Word2Vec berhasil dimuat.")
        except Exception as e:
            print(f"Gagal memuat model: {e}")
    else:
        print(f"File model tidak ditemukan di {MODEL_PATH}. Harap latih model.")

def train_and_save_model(documents_data):
    """
    Melatih model Word2Vec baru dari dokumen yang ada dan menyimpannya.
    'documents_data' adalah dict dari load_documents() di search_engines.py
    """
    global model
    print("Memulai pelatihan model Word2Vec...")
    
    # 1. Siapkan kalimat untuk pelatihan
    # Kita menggunakan teks yang sudah diproses (lowercase, no-stopword, dll)
    sentences = []
    for doc in documents_data.values():
        processed_text = doc.get("processed", "")
        if processed_text:
            # Gensim membutuhkan list dari list of strings
            sentences.append(processed_text.split()) 

    if not sentences:
        print("Tidak ada dokumen untuk dilatih. Pelatihan dibatalkan.")
        return False

    # 2. Latih model Word2Vec
    # vector_size=100 -> 100 dimensi vektor
    # window=5 -> melihat 5 kata sebelum dan 5 kata sesudah
    # min_count=2 -> hanya pertimbangkan kata yang muncul setidaknya 2 kali
    print(f"Melatih model dengan {len(sentences)} dokumen...")
    new_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=2, workers=4)
    
    # 3. Simpan model
    new_model.save(MODEL_PATH)
    print(f"Model berhasil dilatih dan disimpan ke {MODEL_PATH}")
    
    # 4. Perbarui model global di memori
    model = new_model
    return True

def get_similar_words(query, top_n=10):
    """
    Mengkueri model yang sudah dimuat untuk mendapatkan kata-kata serupa.
    """
    global model
    
    # Jika model belum dimuat (misalnya, app baru dimulai & blm dilatih)
    if model is None:
        return [{"word": "Model belum dilatih atau dimuat.", "similarity": 0.0}]

    try:
        query = query.lower().strip()
        # model.wv.most_similar adalah fungsi inti dari Gensim
        similar_results = model.wv.most_similar(query, topn=top_n)
        
        # Format hasil agar sesuai dengan template HTML
        results = [{"word": word, "similarity": float(score)} for word, score in similar_results]
        return results
    
    except KeyError:
        # Ini terjadi jika kata 'query' tidak ada dalam kosakata model
        print(f"Kata '{query}' tidak ditemukan dalam kosakata model.")
        return []
    except Exception as e:
        print(f"Terjadi kesalahan saat mencari kata serupa: {e}")
        return []