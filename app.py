# import dependencies

import chainlit as cl
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration
import csv
import torch

# Initialize the model and tokenizer for embeddings and text generation
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Set padding token for tokenizer
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as the pad_token

# Initialize the T5 model for response generation
response_model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Load FAQ data
faq_data = {}
faq_embeddings = {}
with open('BC 240 QA.csv', mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        question = row['Question'].strip().lower()
        answer = row['Answer'].strip()
        faq_data[question] = answer
        faq_embeddings[question] = None  # Placeholder for precomputed embeddings

# Function to get embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling for single vector
    return embeddings

# Precompute embeddings for FAQ questions
for question in faq_data.keys():
    faq_embeddings[question] = get_embeddings(question)

# Define the chatbot response function
def chatbot_response(user_input):
    user_input = user_input.lower().strip()
    
    # Debug: Print user input
    print(f"User input received: {user_input}")

    # Get embeddings for user input
    user_input_embedding = get_embeddings(user_input)

    # Use cosine similarity to find the best FAQ match
    best_match = None
    best_score = 0

    for question, faq_embedding in faq_embeddings.items():
        score = torch.cosine_similarity(user_input_embedding, faq_embedding).item()

        # Debug: Print similarity score
        print(f"Similarity score for '{question}': {score}")

        if score > best_score:
            best_score = score
            best_match = question

    # Check if the best match is above the threshold
    if best_match and best_score > 0.7:  # Adjust threshold as needed
        print(f"Best match found: {best_match} with score {best_score}")
        return faq_data[best_match]

    # If no match found, generate a response using the T5 model
    inputs = tokenizer("generate: " + user_input, return_tensors='pt', padding=True, truncation=True)
    bot_output = response_model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100)
    response = tokenizer.decode(bot_output[0], skip_special_tokens=True)

    # Debug: Print the generated response
    print(f"Generated response: {response}")
    
    return response

@cl.on_message
async def main(message: str):
    # Get user message
    user_message = message.content if hasattr(message, 'content') else message

    # Debug: Print user message
    print(f"Received message: {user_message}")

    # Generate response
    response = chatbot_response(user_message)

    # Send response back to the user
    await cl.Message(content=response).send()