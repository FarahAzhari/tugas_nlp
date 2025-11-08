# premium_corpus_generator.py
import json
import re
import time
import os
import requests

# Model yang akan digunakan. Anda benar, 2.5 flash adalah rilis stabil.
MODEL_NAME = "gemini-2.5-flash"

# --- 100 Topik AI Unik ---
TOPIC_PROMPTS = [
    # NLP
    "Write a 300-word article about the basics of Natural Language Processing (NLP).",
    "Explain the concept of Sentiment Analysis in 300 words.",
    "What are Large Language Models (LLMs)? Write a 300-word explanation.",
    "Describe the process of Machine Translation and its challenges.",
    "What is Named Entity Recognition (NER) and where is it used?",
    "Explain the difference between tokenization and stemming in NLP.",
    "The role of transformers architecture in modern NLP.",
    "What is text summarization (extractive vs. abstractive)?",
    "Challenges in building effective chatbots and virtual assistants.",
    "The importance of 'context' in language understanding for AI.",

    # Computer Vision
    "Write a 300-word article about the fundamentals of Computer Vision.",
    "Explain Image Classification vs. Object Detection.",
    "What are Convolutional Neural Networks (CNNs) and why are they good for images?",
    "The use of Computer Vision in autonomous vehicles.",
    "How does facial recognition technology work? (300 words)",
    "The role of AI in medical image analysis (e.g., X-rays, MRIs).",
    "What is image segmentation and its applications?",
    "Generative Adversarial Networks (GANs) for image generation.",
    "Challenges in computer vision, such as occlusion and lighting.",
    "The concept of 'feature extraction' in image processing.",

    # Reinforcement Learning
    "Write a 300-word introduction to Reinforcement Learning (RL).",
    "Explain the 'agent-environment' loop in RL.",
    "What is a 'reward signal' and why is it important in RL?",
    "The difference between model-based and model-free RL.",
    "Applications of Reinforcement Learning in robotics and gaming.",
    "What is Q-Learning? A simple explanation.",
    "The exploration vs. exploitation trade-off in RL.",
    "Deep Q-Networks (DQN) and their significance.",
    "Policy Gradients and how they work.",
    "AlphaGo: How Reinforcement Learning conquered the game of Go.",

    # General ML/AI Concepts
    "Explain the difference between Supervised and Unsupervised Learning.",
    "What is a 'neural network'? A 300-word overview.",
    "The concept of 'training data' and 'testing data'.",
    "What is 'overfitting' in machine learning and how to prevent it?",
    "Explain the K-Means clustering algorithm for unsupervised learning.",
    "Decision Trees and Random Forests explained simply.",
    "What is a 'hyperparameter' in machine learning?",
    "The importance of data preprocessing and cleaning.",
    "Bias and Variance trade-off in machine learning.",
    "Support Vector Machines (SVMs): An intuitive explanation.",

    # AI Ethics & Society
    "Write a 300-word article on the importance of AI Ethics.",
    "What is 'algorithmic bias' and how does it occur?",
    "The 'black box' problem and the need for Explainable AI (XAI).",
    "AI's impact on the job market and future of work.",
    "The role of AI in surveillance and privacy concerns.",
    "Regulating AI: Challenges and approaches.",
    "The concept of 'data privacy' in the age of AI.",
    "Ethical considerations in the development of autonomous weapons.",
    "The potential of AI for social good (e.g., climate change, humanitarian aid).",
    "The debate around Artificial General Intelligence (AGI) and superintelligence.",

    # Applications
    "The role of AI in modern drug discovery and personalized medicine.",
    "AI in finance: Algorithmic trading and fraud detection.",
    "How AI is personalizing e-commerce and recommendation engines.",
    "AI's role in supply chain management and logistics optimization.",
    "Smart cities: How AI is used to manage urban infrastructure.",
    "AI in agriculture (precision farming).",
    "The use of AI in cybersecurity for threat detection.",
    "AI in the entertainment industry (e.g., VFX, music composition).",
    "AI for scientific research and accelerating discovery.",
    "How AI is used to improve educational tools and personalized learning.",
    
    # 40 More Diverse Topics
    "The history of Artificial Intelligence, from Turing to today.",
    "What is 'Edge AI' and its advantages?",
    "Federated Learning: Training models without centralizing data.",
    "The role of GPUs and TPUs in deep learning.",
    "What is 'transfer learning' and why is it useful?",
    "AI and its impact on creative industries like art and writing.",
    "The challenges of data annotation and labeling for AI.",
    "What are 'embeddings' in the context of machine learning?",
    "AI in sports analytics and performance optimization.",
    "The concept of 'digital twins' and their simulation power.",
    "Natural Language Generation (NLG) vs. NLU.",
    "AI for accessibility: Helping people with disabilities.",
    "The science of 'prompt engineering' for LLMs.",
    "Quantum Computing and its potential impact on AI.",
    "The difference between AI, Machine Learning, and Deep Learning.",
    "How recommendation systems (like Netflix's) work.",
    "The 'Turing Test' and its relevance today.",
    "AI in space exploration and astronomy.",
    "What is 'swarm intelligence' and where is it applied?",
    "The challenges of building robust AI that can handle uncertainty.",
    "How AI helps in detecting and combating misinformation.",
    "The carbon footprint of training large AI models.",
    "What is 'self-supervised learning'?",
    "The future of human-AI collaboration.",
    "AI's role in predicting natural disasters.",
    "The concept of 'generative AI' beyond just text and images.",
    "Multimodal AI: Understanding text, images, and audio together.",
    "The philosophy of AI: Can machines truly 'think'?",
    "How AI is used in judicial and legal systems.",
    "The future of personalized entertainment with AI.",
    "What is 'adversarial AI' and how to defend against it?",
    "The role of open-source software in AI development.",
    "AI in mental health: Chatbots and diagnostic tools.",
    "The challenges of long-term memory in AI models.",
    "How AI is used to optimize energy grids.",
    "The importance of 'common sense' reasoning for AI.",
    "AI and its application in archaeology and history.",
    "What is 'anomaly detection' and its uses?",
    "The future of robotics and AI integration.",
    "AI's role in managing and understanding climate change data."
]

# --- FUNGSI HELPER BARU UNTUK NAMA FILE ---
def slugify(text):
    """
    Mengubah string menjadi "slug" yang aman untuk nama file.
    Contoh: "A Great Title!" -> "a-great-title"
    """
    if not text:
        return "untitled"
    # 1. Ubah ke huruf kecil
    slug = text.lower()
    # 2. Hapus karakter yang tidak diinginkan (bukan huruf, angka, spasi, atau strip)
    slug = re.sub(r'[^\w\s-]', '', slug)
    # 3. Ganti spasi atau strip berlebih dengan satu strip
    slug = re.sub(r'[\s_]+', '-', slug).strip('-')
    # 4. Batasi panjangnya agar tidak terlalu panjang
    slug = slug[:60].strip('-') # Ambil 60 karakter pertama
    if not slug:
        return "untitled"
    return slug

# --- DIMODIFIKASI: generate_text_from_api ---
def generate_text_from_api(prompt, sec_api_url, retries=5, delay=5):
    """
    Memanggil Gemini API. Sekarang dengan prompt yang dimodifikasi
    untuk meminta JUDUL secara eksplisit (dan dalam Bahasa Inggris).
    """
    
    # --- INI ADALAH PERUBAHAN UTAMA (KEMBALI KE BAHASA INGGRIS) ---
    final_prompt = f"""
    Please follow these instructions EXACTLY:
    1. First, write a single, short, clear TITLE line for the topic below (less than 10 words).
    2. Then, add TWO line breaks (one blank line).
    3. Finally, write a 300-word article based on this topic: '{prompt}'
    
    Your response MUST start with the title on the very first line.
    """
    # ------------------------------------

    payload = {
        "contents": [{"parts": [{"text": final_prompt}]}] # Gunakan final_prompt
    }
    headers = {'Content-Type': 'application/json'}

    for attempt in range(retries):
        try:
            response = requests.post(sec_api_url, headers=headers, data=json.dumps(payload), timeout=90)

            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result:
                    # Kembalikan seluruh teks (Judul \n\n Isi Artikel)
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return f"Error: No content\n\nPrompt was: {prompt}" # Beri judul default

            elif response.status_code == 429:
                print(f"   [!] Rate limit hit. Waiting {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"   [!] API Error: Status {response.status_code}. Waiting {delay}s...")
                print(response.text())
                time.sleep(delay)
                delay *= 2

        except requests.exceptions.RequestException as e:
            print(f"   [!] Request Error: {e}. Waiting {delay}s...")
            time.sleep(delay)
            delay *= 2
            
    return f"Error: Failed to generate\n\nPrompt was: {prompt}" # Beri judul default

# --- DIMODIFIKASI: generate_and_save_premium_files ---
def generate_and_save_premium_files(num_documents, save_directory, api_key):
    """
    Fungsi utama yang dipanggil oleh Flask. Sekarang memisahkan judul
    dan membuat nama file dari judul tersebut.
    """
    
    if not api_key:
        print("Error: API Key tidak diberikan ke premium generator.")
        return 0

    sec_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={api_key}"
    os.makedirs(save_directory, exist_ok=True)
    
    print(f"Starting premium generation of {num_documents} AI articles...")
    
    num_to_generate = min(len(TOPIC_PROMPTS), num_documents)
    if num_to_generate < num_documents:
        print(f"Warning: Diminta {num_documents}, tapi hanya ada {len(TOPIC_PROMPTS)} prompt. Akan membuat {num_to_generate} file.")

    # Tentukan nomor awal file
    prefix = "premium_" # Kita tetap gunakan prefiks ini
    existing_files = [f for f in os.listdir(save_directory) if f.startswith(prefix) and f.endswith(".txt")]
    start_index = 0
    if existing_files:
        numbers = [int(f.split('_')[1]) for f in existing_files if f.split('_')[1].isdigit()]
        if numbers:
            start_index = max(numbers)
    
    print(f"Memulai dari indeks file: {start_index + 1}")
    
    count_generated = 0
    for i in range(num_to_generate):
        prompt = TOPIC_PROMPTS[i]
        print(f"\n[{i+1}/{num_to_generate}] Generating premium doc for: '{prompt[25:60]}...'")
        
        # 1. Panggil API. `doc_content` sekarang berisi: "Judul\n\nIsi..."
        doc_content = generate_text_from_api(prompt, sec_api_url)
        
        # 2. Pisahkan Judul dan Isi
        try:
            # Pisahkan di ganti baris ganda pertama
            title, body = doc_content.split('\n\n', 1) 
            title = title.strip() # Bersihkan spasi
            body = body.strip()
        except ValueError:
            # Jika Gemini gagal (misal mengembalikan error), gunakan fallback
            try:
                # Coba pisah di baris pertama saja
                title, body = doc_content.split('\n', 1)
                title = title.strip()
                body = body.strip()
            except ValueError:
                # Jika gagal total, jadikan judul sebagai error
                title = "Error Generating Title"
                body = doc_content
        
        # 3. Buat nama file dari judul
        current_index = start_index + i + 1
        title_slug = slugify(title) # Ubah "Judul Keren!" -> "judul-keren"
        
        # Nama file baru: "premium_001_judul-keren.txt"
        file_name = f"{prefix}{current_index:03d}_{title_slug}.txt"
        file_path = os.path.join(save_directory, file_name)
        
        # 4. Siapkan konten yang akan disimpan (Judul + Isi)
        content_to_save = f"{title}\n\n{body}"
        
        # 5. Simpan file
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content_to_save)
            print(f"   [OK] Saved: {file_path}")
            count_generated += 1
        except IOError as e:
            print(f"   [X] Error writing file {file_name}: {e}")
            
    print("\n-------------------------")
    print(f"Premium generation complete! {count_generated} files created.")
    return count_generated