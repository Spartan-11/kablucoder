
import nltk
import numpy as np
import random
import string  # to process standard python strings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required nltk packages
nltk.download('punkt')  # tokenizer
nltk.download('wordnet')  # lemmatizer

# Sample corpus (knowledge base)
corpus = """
Hello! I am your chatbot. You can ask me anything about Python.
Python is a high-level, interpreted programming language.
It supports object-oriented, imperative, and functional programming.
You can use Python for web development, data science, AI, and more.
NLTK stands for Natural Language Toolkit.
It is a powerful Python library for text processing and NLP.
To install Python libraries, use pip install <library-name>.
Variables in Python are dynamically typed.
Lists, tuples, dictionaries, and sets are common data structures in Python.
"""

# Text preprocessing
sentence_tokens = nltk.sent_tokenize(corpus)  # list of sentences
lemmatizer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting detection
greeting_inputs = ("hello", "hi", "greetings", "sup", "what's up", "hey")
greeting_responses = ["Hi there!", "Hello!", "Hey!", "Hi! How can I help you?"]

def greeting(sentence):
    """If user input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in greeting_inputs:
            return random.choice(greeting_responses)

# Generate response
def response(user_input):
    user_input = user_input.lower()
    sentence_tokens.append(user_input)

    # Vectorize using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sentence_tokens)

    # Get cosine similarity
    similarity = cosine_similarity(tfidf[-1], tfidf)
    idx = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    sentence_tokens.pop()  # remove user input from sentence list

    if score == 0:
        return "I'm sorry, I don't understand that."
    else:
        return sentence_tokens[idx]

# Main chat loop
def chatbot():
    print("Chatbot: Hello! Ask me anything about Python. Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print("Chatbot: Goodbye! Have a nice day.")
            break
        elif greeting(user_input) is not None:
            print("Chatbot:", greeting(user_input))
        else:
            print("Chatbot:", response(user_input))

# Run chatbot
if __name__ == "__main__":
    chatbot()

