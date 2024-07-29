import pandas as pd
import numpy as np
from pymongo import MongoClient
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
import os
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from pymongo.errors import BulkWriteError

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

client = MongoClient('mongodb+srv://veronikavrvr:Verovero11@testingskripsi1.vqtcefe.mongodb.net/')
db = client['batch_lda']
posts_collection = db['posts']
coherence_collection = db['coherence_scores']

model_directory = os.path.expanduser('~/documents/tugasakhir4/ta4-airflow-dir/lda_model')  # This will use the user's home directory
os.makedirs(model_directory, exist_ok=True)
model_path = os.path.join(model_directory, 'lda_model')

def upload_post(posts):
    valid_data = posts.dropna(subset=['id']).to_dict('records')
    for doc in valid_data:
        if 'id' in doc and not pd.isna(doc['id']):
            doc['_id'] = doc['id']
        else:
            print(f"Skipping document with missing or invalid ID: {doc}")
    if valid_data:  # Ensure we only insert if there is valid data
        try:
            posts_collection.insert_many(valid_data)
            print(f"Data updated successfully in MongoDB! Inserted {len(valid_data)} records.")
        except BulkWriteError as bwe:
            handle_bulk_write_errors(bwe)
    else:
        print("No valid data to insert after merging.")

def handle_bulk_write_errors(bwe):
    for error in bwe.details['writeErrors']:
        if error['code'] == 11000:  # Duplicate key error code
            print(f"Duplicate key error for document with _id: {error['keyValue']['_id']}. Skipping insertion.")
        else:
            print(f"Other bulk write error: {error}")

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

def saving(coherence_lda, lda_model, elapsed_time):
    # coherence
    if np.isnan(coherence_lda):
        print("Calculated coherence score is NaN. Skipping recording.")
    else:
        print(f"Calculated Coherence Score: {coherence_lda}")
        # Record the coherence score with a timestamp
        coherence_score = {
            'coherence_score': coherence_lda,
            'timestamp': pd.Timestamp.now(),  # Record the current time
            'latency' : elapsed_time
        }
        coherence_collection.insert_one(coherence_score)
        print(f"Coherence score recorded: {coherence_score}")
    # lda
    lda_model.save(model_path)
    
if __name__ == '__main__':
    df = pd.read_csv('hot_posts.csv')
    df = df.dropna(subset=['id'])
    upload_post(df)
    os.makedirs(model_directory, exist_ok=True)
    df["full"] = df["title"] + " " + df["body"]
    texts = df['full'].tolist()
    start_time = datetime.now()
    processed_docs = [preprocess_text(doc) for doc in texts]
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]
    dictionary = Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary, passes=10, alpha='auto')
    print("Created new LDA model.")
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary,
                                        coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    saving(coherence_lda, lda_model, elapsed_time.total_seconds())
    print(f"LDA model saved to {model_path} and Coherence posted")