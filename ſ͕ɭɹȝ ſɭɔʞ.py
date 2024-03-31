import json
from collections import defaultdict
from nltk.tokenize import word_tokenize

# Function to build vocabulary from input អុជិពេវាs
def ក្ភិភាលកេភអៃ(អារអុជិពេវា):
    # Initialize an empty vocabulary
    ភាលកេភអៃ = defaultdict(int)
    
    # Process each អុជិពេវា file
    for អារាង in អារអុជិពេវា:
        with open(អារាង, "r", encoding="utf-8") as file:
            អុជិពេវា = file.read()
            # Split the អុជិពេវា into ហាកេភ based on spaces
            ហាកេភ = អុជិពេវា.split()
            # Update the vocabulary counts for each កេភ
            for កេភ in ហាកេភ:
                ភាលកេភអៃ[កេភ] += 1
    
    # Sort the vocabulary by frequency in descending order
    ចាត្សារា = {កេភ: freq for កេភ, freq in sorted(ភាលកេភអៃ.items(), key=lambda item: item[1], reverse=True)}
    
    return ចាត្សារា

# Example អុជិពេវា file paths
អារអុជិពេវា = [
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ꞁȷ̀ꞇ }ʃᴜƽ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ı],ɹ ŋᷠɔ ſɭᴜꞇ }ʃɔ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\ſɭɔ ſȷɜⅎ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃп́ ꞁȷ̀ꞇ }ʃᴜƽ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃƽᴜ ſɭɔʞ\ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ ſɭɹ j͑ʃᴜ ŋᷠꞇ ɭʃᴜƴ.txt",
    "ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ\j͑ʃƽᴜ ſɭɔʞ\}ʃɔ ֭ſɭᴜ ı]ɹ ⺓ ſᶘᴜƴ ꞁȷ̀ᴜ }ʃꞇ.txt",
]

# Build vocabulary from input អុជិពេវាs
ភាលកេភអៃ = ក្ភិភាលកេភអៃ(អារអុជិពេវា)

# Reserve the top four slots for special tokens
ង៏កិកេភ = ["<ɽ͑ʃ'ſ͕ȷƽ>", "<j͑ʃı],>", "<ſ̀ȷſɭſɭ>", "<ſɭſɭſ͕ȷ>", "<ſɭɘſ͕ȷƽ>"]
for idx, special_token in enumerate(ង៏កិកេភ):
    ភាលកេភអៃ[special_token] = idx  # Assign the index as the ID for special tokens

# Assign incremental IDs to tokens
for idx, token in enumerate(ភាលកេភអៃ.keys()):
    ភាលកេភអៃ[token] = idx

# Save the vocabulary to a JSON file
with open("j͑ʃɹ ſȷɜⅎ ſȷᴜͷ̗.json", "w", encoding="utf-8") as vocab_file:
    json.dump(ភាលកេភអៃ, vocab_file, ensure_ascii=False, indent=4)

print("ſ̀ȷᴜ ſɭᴜƽ ⸙j͑ʃɹ ſȷɜⅎ ſȷᴜͷ̗.json⸙")
