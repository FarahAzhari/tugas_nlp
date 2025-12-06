import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==========================================
# 1. DATASET & VOCABULARY (INDONESIA)
# ==========================================

english_sentences = [
    "how are you today",
    "what is your name",
    "good morning",
    "i love ai",
    "hello world",
    "thank you",
    "see you later",
    "artificial intelligence",
    "machine learning is cool"
]

indonesian_sentences = [
    "bagaimana kabarmu hari ini",
    "siapa namamu",
    "selamat pagi",
    "saya suka ai",
    "halo dunia",
    "terima kasih",
    "sampai jumpa nanti",
    "kecerdasan buatan",
    "pembelajaran mesin itu keren"
]

# Token khusus
SOS_token = 0  # Start Of Sentence
EOS_token = 1  # End Of Sentence

def build_vocab(sentences):
    vocab = set()
    for sentence in sentences:
        for word in sentence.split():
            vocab.add(word)
    # Pastikan SOS dan EOS ada di awal
    word2index = {"<SOS>": SOS_token, "<EOS>": EOS_token}
    for word in sorted(list(vocab)):
        if word not in word2index:
            word2index[word] = len(word2index)
    index2word = {i: word for word, i in word2index.items()}
    return word2index, index2word

# Build vocabularies
eng_word2index, eng_index2word = build_vocab(english_sentences)
ind_word2index, ind_index2word = build_vocab(indonesian_sentences)

def sentence_to_tensor(sentence, word2index):
    indexes = [word2index[word] for word in sentence.lower().split() if word in word2index]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long).view(1, -1)

# ==========================================
# 2. MODEL (Seq2Seq)
# ==========================================
class Seq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_tensor, target_tensor):
        # Encoder
        embedded_input = self.encoder(input_tensor)
        _, (hidden, cell) = self.rnn(embedded_input)

        # Decoder
        embedded_target = self.decoder(target_tensor)
        output, _ = self.rnn(embedded_target, (hidden, cell))
        output = self.fc(output)
        return output

    def predict(self, input_tensor):
        # 1. Encode
        with torch.no_grad():
            embedded_input = self.encoder(input_tensor)
            _, (hidden, cell) = self.rnn(embedded_input)

        # 2. Decode
        decoded_words = []
        decoder_input = torch.tensor([[SOS_token]], dtype=torch.long)

        for _ in range(15): # Max panjang kalimat
            with torch.no_grad():
                embedded_input = self.decoder(decoder_input)
                output, (hidden, cell) = self.rnn(embedded_input, (hidden, cell))
                prediction = self.fc(output)
                
                topv, topi = prediction.topk(1)
                idx = topi.item()

                if idx == EOS_token:
                    break
                
                decoded_words.append(ind_index2word[idx])
                decoder_input = topi.detach().view(1, -1)

        return decoded_words

# ==========================================
# 3. TRAINING (BAGIAN YANG DIPERBAIKI)
# ==========================================
input_size = len(eng_word2index)
output_size = len(ind_word2index)
hidden_size = 256

model = Seq2Seq(input_size, output_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def train_model():
    print("Melatih model Seq2Seq (English -> Indonesia)...")
    model.train()
    epochs = 300
    
    for epoch in range(epochs):
        for i in range(len(english_sentences)):
            input_tensor = sentence_to_tensor(english_sentences[i], eng_word2index)
            target_tensor = sentence_to_tensor(indonesian_sentences[i], ind_word2index)

            # --- PERBAIKAN DI SINI ---
            # Kita harus menambahkan SOS di DEPAN target untuk input decoder
            # target_tensor asli: [terima, kasih, EOS]
            # decoder_input baru: [SOS, terima, kasih]
            
            sos_tensor = torch.tensor([[SOS_token]], dtype=torch.long)
            
            # Gabungkan SOS + target (kecuali EOS terakhir)
            decoder_input = torch.cat((sos_tensor, target_tensor[:, :-1]), dim=1)
            
            optimizer.zero_grad()
            
            # Forward pass dengan decoder_input yang benar
            output = model(input_tensor, decoder_input)
            
            # Target untuk loss adalah kalimat asli [terima, kasih, EOS]
            # Kita membandingkan prediksi langkah 1 dengan 'terima', langkah 2 dengan 'kasih', dst.
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = target_tensor.contiguous().view(-1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    print("Training Selesai! Model siap.")

# Train saat start
train_model()

# ==========================================
# 4. FUNGSI TRANSLATE
# ==========================================
def simple_translate(text):
    model.eval()
    try:
        processed_text = text.lower().strip()
        # Hilangkan tanda tanya/seru biar simple match
        import string
        processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))
        
        # Cek vocab
        for word in processed_text.split():
            if word not in eng_word2index:
                return f"Maaf, kata '{word}' tidak ada di dataset latih."

        input_tensor = sentence_to_tensor(processed_text, eng_word2index)
        result_words = model.predict(input_tensor)
        return " ".join(result_words)
    except Exception as e:
        return f"Error: {str(e)}"