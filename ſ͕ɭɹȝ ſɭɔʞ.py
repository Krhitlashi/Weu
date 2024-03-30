import json
from collections import defaultdict
from nltk.tokenize import word_tokenize

# Function to build vocabulary from input texts
def build_vocab_from_texts(file_paths):
    # Initialize an empty vocabulary
    vocab = defaultdict(int)
    
    # Process each text file
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            # Split the text into words based on spaces
            words = text.split()
            # Update the vocabulary counts for each word
            for word in words:
                vocab[word] += 1
    
    # Sort the vocabulary by frequency in descending order
    sorted_vocab = {word: freq for word, freq in sorted(vocab.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_vocab

# Example text file paths
file_paths = [
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ꞁȷ̀ꞇ }ʃᴜƽ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃƽᴜ ſɭɔʞ\ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃƽᴜ ſɭɔʞ\}ʃɔ ֭ſɭᴜ ı]ɹ ⺓ ſᶘᴜƴ ꞁȷ̀ᴜ }ʃꞇ.txt",
]

# Build vocabulary from input texts
vocab = build_vocab_from_texts(file_paths)

# Reserve the top four slots for special tokens
special_tokens = ["<ɽ͑ʃ'ſ͕ȷƽ>", "<j͑ʃı],>", "<ſ̀ȷſɭſɭ>", "<ſɭſɭſ͕ȷ>", "<ſɭɘſ͕ȷƽ>"]
for idx, special_token in enumerate(special_tokens):
    vocab[special_token] = idx  # Assign the index as the ID for special tokens

# Assign incremental IDs to tokens
start_id = 0  # Start assigning IDs after the special tokens
for idx, token in enumerate(vocab.keys()):
    vocab[token] = start_id + idx

# Save the vocabulary to a JSON file
with open("j͑ʃɹ ſȷɜⅎ ſȷᴜͷ̗.json", "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)

print("Vocabulary created and saved to j͑ʃɹ ſȷɜⅎ ſȷᴜͷ̗.json")
