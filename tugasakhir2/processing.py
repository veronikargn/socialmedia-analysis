from kafka import KafkaConsumer
import logging
import joblib
from pymongo import MongoClient
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
import json
from datetime import datetime

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# MongoDB setup
client = MongoClient('MONGODBCONNECTION')
db = client['realtime_lsa']
coherence_collection = db['coherence_scores']
post_collection = db['posts']

# Paths for saved models
current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'lsa_model', 'lsa_model.pkl')
vectorizer_path = os.path.join(current_directory, 'lsa_model', 'tfidf_vectorizer.pkl')

# Kafka consumer setup
consumer = KafkaConsumer('reddit_updates', bootstrap_servers='kafka:9092', value_deserializer=lambda m: json.loads(m.decode('utf-8')))

# Load stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Filter out stop words and apply lemmatization
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
        return filtered_tokens
    return []

def update_lsa_model(title, body):
    # Combine title and body into a single text
    new_document = title + " " + body
    data = pd.DataFrame(post_collection.find())
    data["full"] = data["title"] + " " + data["body"]
    texts= data['full'].tolist()
    
    start_time = datetime.now()
    # Preprocess the new document
    processed_new_doc = preprocess_text(new_document)
    
    # Ensure the processed document is not empty
    if len(processed_new_doc) == 0:
        logging.warning("Processed new document is empty after preprocessing.")
        return
    
    # Convert preprocessed document to a format suitable for TF-IDF
    new_tfidf_matrix = vectorizer.transform([" ".join(processed_new_doc)])
    
    # Apply the loaded LSA model to transform the new document
    svd_model.transform(new_tfidf_matrix)
    
    # Get feature names (terms) from the TF-IDF matrix
    terms = vectorizer.get_feature_names_out()
    
    # Extract topics by finding the top words in each component
    new_lsa_topics = []
    for idx, topic in enumerate(svd_model.components_):
        topic_terms = [terms[i] for i in topic.argsort()[:-10 - 1:-1]]
        new_lsa_topics.append(topic_terms)
    
    processed_docs = [preprocess_text(doc) for doc in texts]
    # Filter out empty documents
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]

    # Create a dictionary and corpus
    dictionary = Dictionary(processed_docs)
    
    # Calculate coherence score using gensim's CoherenceModel
    coherence_model = CoherenceModel(topics=new_lsa_topics, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    
    # Calculate elapsed time
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    
    # Save coherence score to MongoDB
    coherence_entry = {
        'coherence_score': coherence_score,
        'timestamp': pd.Timestamp.now(),  # Record the current time
        'latency': elapsed_seconds
    }
    coherence_collection.insert_one(coherence_entry)
    print("Coherence score saved to MongoDB.")

# Kafka consumer message processing
for message in consumer:
    logging.info(f"Received message from Kafka: {message.value}")
    if message.value.get("event") == "new_post":
        try:
            vectorizer = joblib.load(vectorizer_path)
            svd_model = joblib.load(model_path)
            
            # Extract title and body from the Kafka message
            title = message.value.get("title", "")
            body = message.value.get("body", "")
            
            # Update LSA model with the extracted data
            update_lsa_model(title, body)
            
        except Exception as e:
            logging.error(f"Error updating LSA model: {e}")
