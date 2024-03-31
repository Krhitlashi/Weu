import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

# Load the custom pretrained language model
អារាវេំ = "ᶅſɔⅎ.pt"  # Path to your custom trained model
កឺត្សុ = GPT2Config.from_json_file("ᶅſɔⅎ.json") # Path to your configuration file
វេំ = GPT2LMHeadModel.from_pretrained(អារាវេំ, config=កឺត្សុ)

# Load the custom tokenizer
ចាថេសអេស្កេក = GPT2Tokenizer.from_pretrained("ı],ᴜ ſ͕ɭᴜ j͑ʃᴜꞇ ꞁȷ̀ɔ j͑ʃƽɔƽ")

# Define a list of special tokens to remove
ង៏កិ១សៃអេស្កេក = [
    "<ɽ͑ʃ'ſ͕ȷƽ>",
    "<ſɭſɭſ͕ȷ>",
    "<j͑ʃı],>",
    "<ſ̀ȷſɭſɭ>",
    "<ſɭɘſ͕ȷƽ>",
]

ភាលកេភអៃ = ចាថេសអេស្កេក.get_vocab()

# ꞁȷ̀ɹ ʃᴜ ſɭɹ ſןɹ
ហាសាភេនី = [
    "ꞁȷ̀ɔ ſᶘᴜ ɽ͑ʃ'ᴜ ɭʃꞇȝ ſɟɔƴ ſɭɔƴ",
    "ɭʃᴜ ꞁȷ̀ᴜ ɽ͑ʃ'ᴜȝ ſɭɔƴ",
    "ꞁȷ̀ɜ j͐ʃɹ ŋᷠꞇ",
    "}ʃɔ ֭ſɭᴜ ı],ɹ ɭʃᴜȝ"
    ]

# Store generated sequences in a list
ចាត្សារា = []

# Define a function to decode token IDs using the custom vocabulary
def decode_token_ids(encoded_list):
    # Create a reverse mapping of token IDs to tokens in your custom vocabulary
    reverse_vocab = {v: k for k, v in ភាលកេភអៃ.items()}
    # Filter out token IDs that are not in the custom vocabulary
    decoded_tokens = [reverse_vocab[token_id] for token_id in encoded_list if token_id in reverse_vocab]
    # Join the tokens together to form the decoded text
    decoded_text = ' '.join(decoded_tokens)
    
    return decoded_text

# Encoding and Generation
def ថេខាវេកិភ(អុជិភេវា):
    # Encode Single Input Text
    input_ids = ចាថេសអេស្កេក.encode(
        អុជិភេវា,
        add_special_tokens=True,
        return_tensors='pt',
    )

    # Convert input_ids to PyTorch tensor
    រឺថា = input_ids.clone().detach()

    # Generate text for the prompt
    ត្សេំនី = វេំ.generate(
        រឺថា,
        max_length=200,
        num_return_sequences=1,  # Number of text samples to generate
        no_repeat_ngram_size=2,  # Avoid repeating the same phrase
        do_sample=True,
        top_p=0.8,  # Adjust the top-p value for randomness
        temperature=0.2,  # Adjust the temperature for creativity
    )

    # Append generated sequences to the list
    ចាត្សារា.extend(ត្សេំនី.tolist())

    # Decode the generated text using decode_token_ids function
    generated_token_ids = ត្សេំនី[0].tolist()
    generated_text = decode_token_ids(generated_token_ids)
    return generated_text

# ꞁȷ̀ɹ ʃᴜ ſɭɹ ſןɹ
for សាភេនី in ហាសាភេនី:
    ថេខាវេកិភ(សាភេនី)

# Decode the generated sequences
decoded_sequences = []
for រឺថា in ចាត្សារា:
    if រឺថា:
        # Decode the token IDs
        decoded_text = decode_token_ids(រឺថា)
    else:
        decoded_text = "ꞁȷ̀ɔȝ ɭl̀ɹƽ"  # Handle empty sequences

    decoded_sequences.append(decoded_text)

for j, decoded_text in enumerate(decoded_sequences):
    for special_token in ង៏កិ១សៃអេស្កេក:
        decoded_text = decoded_text.replace(special_token, "")
    print(f"|{j} j͑ʃᴜꞇ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ|")
    print(decoded_text)
    print("")

# Interactive input and generation loop ı],ɹ ſɭ,ɹ
while True:
    print("j͑ʃɹ ʃᴜ }ʃᴜ ſɭɔʞ ꞁȷ̀ɔƽ ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ |ꞁȷ̀ᴜ ŋᷠɹ ꞁȷ̀ɔƽ j͑ʃɹ ʃᴜ }ʃᴜ ⸙ ⸙|")
    ល៏មារ = input()
    
    if ល៏មារ.lower() == ' ':
        break
    if ល៏មារ.lower() == '':
        break
    
    ត្លាកាកល៏មារ = ថេខាវេកិភ(ល៏មារ)
    for special_token in ង៏កិ១សៃអេស្កេក:
        ត្លាកាកល៏មារ = decoded_text.replace(special_token, "")
    print("")
    print("|ꞁȷ̀ɜ ı],ɹ ſןɔ ᶅſᴜ|")
    print(ត្លាកាកល៏មារ)
    print("")