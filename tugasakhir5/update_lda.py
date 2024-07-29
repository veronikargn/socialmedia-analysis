from pymongo import MongoClient
from kafka import KafkaConsumer
import logging
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import pandas as pd
import os
from datetime import datetime
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

client = MongoClient('MONGODBCONNECTION')
db = client['realtime_lda']
posts_collection = db['posts']
coherence_collection = db['coherence_scores']

current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, 'lda_model', 'lda_model')

consumer = KafkaConsumer('reddit_updates', bootstrap_servers='kafka:9092', value_deserializer=lambda m: json.loads(m.decode('utf-8')))

def preprocess_text(text):
    if isinstance(text, str):  # Ensure the text is a string
        # Tokenize the text
        tokens = word_tokenize(text.lower())
        # Load stop words
        stop_words = set(stopwords.words('english'))
        # Initialize lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Filter out stop words and apply lemmatization
        filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalnum()]
        return filtered_tokens
    return []

def download_existing_data():
    logging.info("Downloading existing data from MongoDB...")
    existing_data = list(posts_collection.find())
    if existing_data:
        logging.info(f"Downloaded {len(existing_data)} existing posts from MongoDB.")
        return pd.DataFrame(existing_data)
    else:
        logging.info("No existing data found in MongoDB.")
        return pd.DataFrame()
    
def saving(coherence_lda, lda_model, elapsed_time):
    # coherence
    if np.isnan(coherence_lda):
        logging.warning("Calculated coherence score is NaN. Skipping recording.")
    else:
        logging.info(f"Calculated Coherence Score: {coherence_lda}")
        # Record the coherence score with a timestamp
        coherence_score = {
            'coherence_score': coherence_lda,
            'timestamp': pd.Timestamp.now(),  # Record the current time
            'latency' : elapsed_time
        }
        coherence_collection.insert_one(coherence_score)
        logging.info(f"Coherence score recorded: {coherence_score}")
    # lda
    lda_model.save(model_path)

def update_lda_model():
    data = download_existing_data()  
    logging.info(f"Downloaded {len(data)} existing posts for preprocessing.")
    data["full"] = data["title"] + " " + data["body"]  
    texts = data['full'].tolist()
    lda_model = LdaModel.load(model_path)
    logging.info("Loaded existing LDA model.")
    # Preprocess texts
    start_time = datetime.now()
    dictionary = lda_model.id2word
    processed_docs = [preprocess_text(doc) for doc in texts]
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]
# Filter the processed documents to contain only words in the dictionary
    filtered_docs = []
    valid_words = set(dictionary.token2id.keys())
    for doc in processed_docs:
        filtered_doc = [word for word in doc if word in valid_words]
        if filtered_doc:  # Only keep non-empty documents
            filtered_docs.append(filtered_doc)

# Create the corpus using the filtered documents
    corpus = [dictionary.doc2bow(doc) for doc in filtered_docs]
    # Update the LDA model with the filtered corpus
    lda_model.update(corpus, passes=10, iterations=50)
    lda_model.sync_state()
    logging.info("Updated existing LDA model with new data.")

    coherence_model_lda = CoherenceModel(model=lda_model, texts=filtered_docs, dictionary=lda_model.id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    saving(coherence_lda, lda_model, elapsed_time.total_seconds())

for message in consumer:
    logging.info(f"Received message from Kafka: {message.value}")
    if message.value.get("event") == "new_post":
        try:
            update_lda_model()
        except Exception as e:
            logging.error(f"Error updating LDA model: {e}")