#Import dependencies

import chainlit as cl
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import csv
import torch

# Initialize the model and tokenizer for embeddings
embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Add a new pad token
embed_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
embedding_model.resize_token_embeddings(len(embed_tokenizer))

# Initialize the tokenizer and model for text generation
gen_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
gen_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Set padding token for generation tokenizer
gen_tokenizer.pad_token = gen_tokenizer.eos_token

# Load FAQ data
faq_data = {}
with open('BC 240 QA.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        question = row['Question'].strip().lower()
        answer = row['Answer'].strip()
        faq_data[question] = answer

# Precompute embeddings for FAQ questions
faq_embeddings = {}
for question in faq_data.keys():
    faq_embeddings[question] = None  # Placeholder for embeddings

def get_embeddings(text):
    inputs = embed_tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Compute embeddings for FAQs
for question in faq_data.keys():
    faq_embeddings[question] = get_embeddings(question)

def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    print(f"User input received: {user_input}")

    # Get embeddings for user input
    user_input_embedding = get_embeddings(user_input)

    # Use cosine similarity to find the best FAQ match
    best_match = None
    best_score = -1

    for question, faq_embedding in faq_embeddings.items():
        score = torch.cosine_similarity(user_input_embedding, faq_embedding).item()
        print(f"Similarity score for '{question}': {score}")

        if score > best_score:
            best_score = score
            best_match = question

    # Check if the best match is above the threshold
    if best_match and best_score > 0.7:
        print(f"Best match found: {best_match} with score {best_score}")
        return faq_data[best_match]

    # If no match found, generate a response using the text generation model
    inputs = gen_tokenizer.encode(user_input + gen_tokenizer.eos_token, return_tensors='pt', padding=True, truncation=True)
    outputs = gen_model.generate(inputs, max_length=1500, pad_token_id=gen_tokenizer.eos_token_id)
    
    # Explicitly set clean_up_tokenization_spaces to avoid the warning
    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"Generated response: {response}")

    return response

@cl.on_message
async def main(message):
    # Get user message
    user_message = message.content if hasattr(message, 'content') else message
    print(f"Received message: {user_message}")

    # Generate response
    response = chatbot_response(user_message)

    # Send response back to the user
    await cl.Message(content=response).send()
