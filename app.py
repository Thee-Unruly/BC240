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