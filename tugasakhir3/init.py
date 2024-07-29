import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
import os
import joblib
import numpy as np
from datetime import datetime

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


client = MongoClient('mongodb+srv://veronikavrvr:Verovero11@testingskripsi1.vqtcefe.mongodb.net/')
db = client['batch_lsa']
posts_collection = db['posts']
coherence_collection = db['coherence_scores']

model_directory = os.path.expanduser('~/documents/tugasakhir3/ta3-airflow-dir/lsa_model')  # This will use the user's home directory
os.makedirs(model_directory, exist_ok=True)
model_path = os.path.join(model_directory, 'lsa_model.pkl')
vectorizer_path = os.path.join(model_directory, 'tfidf_vectorizer.pkl')

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

def saving(coherence_lsa, svd_model, vectorizer,elapsed_time):
        # coherence
    if np.isnan(coherence_lsa):
        print("Calculated coherence score is NaN. Skipping recording.")
    else:
        print(f"Calculated Coherence Score: {coherence_lsa}")
            # Record the coherence score with a timestamp
        coherence_score = {
            'coherence_score': coherence_lsa,
            'timestamp': pd.Timestamp.now(),  # Record the current time
            'latency' : elapsed_time
        }
        try:
            coherence_collection.insert_one(coherence_score)
            print(f"Coherence score recorded: {coherence_lsa}")
        except Exception as e:
            print(f"Error saving coherence score: {e}")
        # lda
    try:
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(svd_model, model_path)
        print(f"LSA model saved to {model_path}")
    except Exception as e:
        print(f"Error saving LSA model: {e}")

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

# Preprocess text
stop_words = stopwords.words('english')

if __name__ == '__main__':
    df=pd.read_csv('hot_posts.csv')
    df["full"] = df["title"] + " " + df["body"]
    documents = df['full'].tolist()

    start_time = datetime.now()
    # Apply preprocessing and filter out non-string documents
    processed_docs = [preprocess_text(doc) for doc in documents]

# Filter out empty documents that resulted from non-string or stopword-only inputs
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]

# Create a dictionary and corpus
    dictionary = Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Convert documents to TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(doc) for doc in processed_docs])

# Apply LSA using TruncatedSVD from sklearn
    n_topics = 5
    svd_model = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_matrix = svd_model.fit_transform(tfidf_matrix)

# Get feature names (terms) from the TF-IDF matrix
    terms = vectorizer.get_feature_names_out()

# Extract topics by finding the top words in each component
    lsa_topics = []
    for idx, topic in enumerate(svd_model.components_):
        topic_terms = [terms[i] for i in topic.argsort()[:-10 - 1:-1]]
        lsa_topics.append(topic_terms)

    # Print the topics
    for i, topic in enumerate(lsa_topics):
        print(f"Topic {i + 1}: {topic}")

    # Calculate coherence score using gensim's CoherenceModel
    coherence_model = CoherenceModel(topics=lsa_topics, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    saving(coherence_score, svd_model, vectorizer, elapsed_seconds)
    upload_post(df)
    print(f"\nCoherence Score: {coherence_score}")