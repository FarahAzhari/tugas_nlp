# ai_corpus_generator.py
import os
import random

# --- Data Bank ---
TOPICS = {
    # ... (Semua data TOPICS Anda tetap di sini, tidak berubah) ...
    "NLP": [
        "Natural Language Processing (NLP) is a subfield of AI focused on the interaction between computers and humans using natural language.",
        "Core NLP tasks include text classification, sentiment analysis, machine translation, and question answering. These tasks power everything from chatbots to search engines.",
        "Large Language Models (LLMs) like GPT and BERT have revolutionized NLP. They are trained on massive datasets of text and can generate human-like prose, translate languages, and summarize complex information.",
        "The challenges in NLP include understanding context, ambiguity, and the nuances of human communication, such as sarcasm or irony. Building models that truly 'understand' is the next frontier."
    ],
    "Computer_Vision": [
        "Computer Vision (CV) enables computers to 'see' and interpret visual information from the world, such as images and videos.",
        "Applications of CV are widespread, including facial recognition for security, autonomous vehicle navigation, and medical image analysis (e.g., detecting tumors in X-rays).",
        "Convolutional Neural Networks (CNNs) are the dominant architecture used in computer vision. They are designed to automatically and adaptively learn spatial hierarchies of features, from edges to complex objects.",
        "Challenges in CV involve handling occlusions (partially hidden objects), variations in lighting, and the sheer amount of data required for training robust models."
    ],
    "Reinforcement_Learning": [
        "Reinforcement Learning (RL) is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize cumulative reward.",
        "RL is famous for its success in games, such as AlphaGo defeating the world champion in Go. However, its real-world applications are growing in robotics, finance, and resource management.",
        "The core components of an RL system are the agent, the environment, a state, a set of actions, and a reward signal. The agent learns through trial and error.",
        "A major challenge in RL is the 'sample-efficiency' problem—it often takes millions of trials for an agent to learn an optimal policy, which can be impractical in the real world."
    ],
    "AI_Ethics": [
        "AI Ethics explores the moral implications and societal impact of artificial intelligence. As AI becomes more powerful, ensuring it is used responsibly is critical.",
        "Key ethical concerns include algorithmic bias, which can perpetuate or even amplify existing human prejudices in areas like hiring, lending, and criminal justice.",
        "Transparency and 'explainability' (XAI) are major topics. If an AI model makes a critical decision (e.g., a medical diagnosis), humans need to be able to understand *why* it made that choice.",
        "The 'alignment problem' is a long-term concern: how do we ensure that the goals of highly advanced AI systems remain aligned with human values? This is a foundational question for AI safety."
    ],
    "Robotics": [
        "Robotics integrates AI, mechanical engineering, and computer science to design and build intelligent machines that can interact with the physical world.",
        "AI powers robotic perception (using sensors and computer vision to understand surroundings) and motion planning (determining how to move to complete a task).",
        "Modern robotics is moving beyond fixed automation in factories. Collaborative robots ('cobots') are designed to work safely alongside humans, and autonomous systems are being developed for logistics and delivery.",
        "Sim-to-real transfer is a key technique where robotic agents are trained in a physics simulation before being deployed on a real robot, saving time and preventing costly hardware damage."
    ],
    "General_AI": [
        "Artificial General Intelligence (AGI) refers to a hypothetical type of AI that possesses the ability to understand, learn, and apply its intelligence to solve any problem, much like a human being.",
        "Current AI systems are considered 'narrow AI'—they are designed for specific tasks, like playing chess or translating language. AGI would be a system with generalized cognitive abilities.",
        "The path to AGI is highly debated. Some researchers believe it will emerge from scaling up current models, while others argue that entirely new architectures and paradigms are required.",
        "The development of AGI would represent a profound technological leap, bringing immense benefits but also significant risks that must be carefully managed by the AI safety community."
    ],
    "Data_Science": [
        "Data Science is an interdisciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
        "Key components of Data Science include statistics, machine learning, data visualization, and domain expertise.",
        "A data scientist cleans, analyzes, and interprets complex data, while a data engineer builds the pipelines that transport this data.",
        "Popular tools in Data Science include Python, R, SQL, and libraries like Pandas, Scikit-learn, and TensorFlow."
    ],
    "Machine_Learning": [
        "Machine Learning (ML) is the foundation of modern AI, where systems learn patterns from data without being explicitly programmed.",
        "Supervised learning involves training a model on labeled data, where both the input and the desired output are known. This is used for classification and regression.",
        "Unsupervised learning finds hidden patterns in unlabeled data. Clustering and dimensionality reduction are common unsupervised tasks.",
        "The typical ML workflow involves data collection, data preprocessing, model training, evaluation, and deployment. This iterative process is key to building effective models."
    ],
    "Deep_Learning": [
        "Deep Learning is a specialized subfield of Machine Learning that uses artificial neural networks with many layers (hence 'deep').",
        "Neural networks are inspired by the human brain, composed of interconnected nodes or 'neurons' in layered structures.",
        "Deep Learning has driven breakthroughs in complex tasks, especially in computer vision with CNNs and in language with Recurrent Neural Networks (RNNs) and Transformers.",
        "Training deep models requires large amounts of data and significant computational power, often relying on GPUs (Graphics Processing Units) to perform calculations in parallel."
    ],
    "Generative_AI": [
        "Generative AI refers to models that can create new content, such as text, images, music, or code, rather than just analyzing existing data.",
        "Generative Adversarial Networks (GANs) were a major breakthrough, using two competing neural networks (a generator and a discriminator) to create highly realistic images.",
        "Diffusion models are another popular technique, which work by progressively adding noise to an image until it's pure static, and then training a model to reverse the process.",
        "Large Language Models are a form of Generative AI focused on text. They power applications like advanced chatbots, code assistants, and creative writing tools."
    ],
    "Expert_Systems": [
        "Expert Systems were an early form of AI popular in the 1980s. They are designed to mimic the decision-making ability of a human expert in a narrow domain.",
        "These systems consist of two main parts: a knowledge base (containing facts and 'if-then' rules) and an inference engine (which applies the rules to new data).",
        "While less common today, the principles of expert systems are still found in rule-based business logic, diagnostic tools, and financial software.",
        "The limitation of expert systems is their 'brittleness'—they cannot handle problems outside their specific programmed knowledge base and are difficult to update."
    ],
    "Speech_Recognition": [
        "Speech Recognition, or Automatic Speech Recognition (ASR), is a technology that converts spoken language into written text.",
        "Modern ASR systems use Deep Learning models, particularly RNNs and Transformers, to handle the complexities of human speech, including different accents and background noise.",
        "This technology is the core of virtual assistants like Siri and Alexa, as well as live transcription services and voice-controlled systems.",
        "Challenges in speech recognition include understanding context for ambiguous words (e.g., 'write' vs. 'right') and performing well in 'far-field' scenarios where the speaker is distant from the microphone."
    ],
}


# --- Logika Markov Chain ---

def build_markov_model(all_words, n_gram=2):
    """
    Membangun model Markov Chain dan daftar 'sentence_starters'.
    """
    model = {}
    sentence_starters = [] # <-- PERUBAHAN
    
    if len(all_words) < n_gram:
        return model, sentence_starters # Kembalikan tuple kosong

    # --- PERUBAHAN: Selalu anggap pasangan kata pertama sebagai starter ---
    sentence_starters.append(tuple(all_words[0 : n_gram]))
    
    for i in range(len(all_words) - n_gram):
        prefix = tuple(all_words[i : i + n_gram])
        suffix = all_words[i + n_gram]
        
        # --- PERUBAHAN: Cek apakah ini awal kalimat ---
        # Jika kata *sebelumnya* diakhiri dengan titik/tanda tanya/seru
        if i > 0 and all_words[i-1].endswith(('.', '?', '!')):
            sentence_starters.append(prefix)
        # ----------------------------------------------
            
        if prefix not in model:
            model[prefix] = []
        model[prefix].append(suffix)
        
    print(f"Model Markov Chain dibuat dengan {len(model)} prefiks dan {len(sentence_starters)} permulaan kalimat.")
    # --- PERUBAHAN: Kembalikan model DAN starters ---
    return model, sentence_starters

def generate_markov_text(model, sentence_starters, n_gram=2, min_length=100, max_length=200):
    """
    Menghasilkan teks baru menggunakan model Markov.
    """
    # --- PERUBAHAN: Memerlukan sentence_starters ---
    if not model or not sentence_starters:
        return "Model is empty. Cannot generate text."
        
    # --- PERUBAHAN: Mulai HANYA dari starter yang valid ---
    current_key = random.choice(sentence_starters)
    output_words = list(current_key)
    
    for _ in range(max_length - n_gram):
        if current_key not in model:
            # Jika buntu, mulai lagi dari awal kalimat (bukan acak)
            current_key = random.choice(sentence_starters)
            # Hindari penggabungan kalimat yang aneh
            output_words.append(".") 
            output_words.extend(list(current_key))
            
        next_word = random.choice(model[current_key])
        output_words.append(next_word)
        current_key = tuple(output_words[-n_gram:])
        
        if len(output_words) > min_length and next_word.endswith(('.', '?', '!')):
            break
    
    # --- PERUBAHAN: Pastikan huruf pertama adalah kapital ---
    if output_words:
        output_words[0] = output_words[0].capitalize()
            
    return " ".join(output_words)


# --- Fungsi Analisis Judul (Tidak Berubah) ---
def find_main_topic_from_text(body_text):
    topic_keywords = {}
    for key in TOPICS.keys():
        keywords = [key.lower()]
        keywords.append(key.replace("_", " ").lower())
        topic_keywords[key] = keywords

    body_lower = body_text.lower()
    scores = {}
    
    for topic_name, keywords in topic_keywords.items():
        count = 0
        for kw in keywords:
            count += body_lower.count(kw)
        scores[topic_name] = count
        
    if not scores:
        return None
    
    best_topic = max(scores, key=scores.get)
    
    if scores[best_topic] == 0:
        return None
            
    return best_topic


# --- FUNGSI UTAMA (DIMODIFIKASI) ---

def generate_and_save_files(num_documents, save_directory):
    """
    Fungsi utama yang dipanggil oleh Flask untuk membuat file.
    """
    
    # 1. Siapkan Data Pelatihan
    print("Mempersiapkan data pelatihan untuk model Markov...")
    all_text = " ".join(p for topic_paras in TOPICS.values() for p in topic_paras)
    all_words = all_text.split()
    
    if len(all_words) < 3:
        print("Data pelatihan tidak cukup. Batalkan pembuatan.")
        return 0

    # 2. Latih Model (SEKARANG MENGEMBALIKAN 2 NILAI)
    markov_model, sentence_starters = build_markov_model(all_words, n_gram=2)
    
    if not markov_model or not sentence_starters:
         print("Model Markov gagal dibuat (data tidak cukup).")
         return 0
    
    # 3. Tentukan nomor awal file
    # ... (kode ini tidak berubah) ...
    os.makedirs(save_directory, exist_ok=True)
    existing_files = [f for f in os.listdir(save_directory) if f.startswith("doc_") and f.endswith(".txt")]
    start_index = 0
    if existing_files:
        numbers = [int(f[4:-4]) for f in existing_files if f[4:-4].isdigit()]
        if numbers:
            start_index = max(numbers)
    
    print(f"Memulai pembuatan {num_documents} file dari indeks {start_index + 1}...")
    
    count_generated = 0
    for i in range(num_documents):
        current_index = start_index + i + 1
        
        # 4. Generate Teks (SEKARANG MENGGUNAKAN sentence_starters)
        body = generate_markov_text(markov_model, sentence_starters, n_gram=2, min_length=150, max_length=300)
        
        # 5. Analisis Teks untuk Judul (Tidak berubah)
        main_topic_key = find_main_topic_from_text(body)
        
        if main_topic_key:
            title_text = main_topic_key.replace("_", " ") 
            title = f"A Generated Analysis on {title_text}"
        else:
            title = f"Generative Document {current_index}"
        
        doc_content = f"{title}\n\n{body}"
        
        # 6. Simpan File (Tidak berubah)
        # ... (kode ini tidak berubah) ...
        file_name = f"doc_{current_index:03d}.txt"
        file_path = os.path.join(save_directory, file_name)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_content)
            count_generated += 1
        except IOError as e:
            print(f"Error writing file {file_name}: {e}")
            
    print(f"\nGeneration complete! {count_generated} new files created.")
    return count_generated