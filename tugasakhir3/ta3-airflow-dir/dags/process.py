import pandas as pd
import logging
from pymongo import MongoClient
import praw
from gensim.models import CoherenceModel
import joblib
import nltk
from gensim.corpora.dictionary import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import pendulum
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from datetime import datetime

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Reddit API configuration
reddit = praw.Reddit(
    client_id="CLIENT_ID",
    client_secret="CLIENT_SECRET",
    user_agent="USER_AGENT",
)
subreddit_name = 'beauty'
subreddit = reddit.subreddit(subreddit_name)

client = MongoClient('MONGODBCONNECTION')
db = client['batch_lsa']
posts_collection = db['posts']
coherence_collection = db['coherence_scores']

model_directory = '/opt/airflow/lsa_model'
model_path = os.path.join(model_directory, 'lsa_model.pkl')
vectorizer_path = os.path.join(model_directory, 'tfidf_vectorizer.pkl')

def upload_post(posts):
    valid_data = posts.dropna(subset=['id']).to_dict('records')
    for doc in valid_data:
        # Ensure _id is set correctly for MongoDB uniqueness
        doc['_id'] = doc['id']  
        # Insert updated data with unique IDs
    if valid_data:  # Ensuring only valid data is inserted
        posts_collection.insert_many(valid_data)
        logging.info(f"Data updated successfully in MongoDB! Inserted {len(valid_data)} records.")
    else:
        logging.info("No valid data to insert after merging.")
    
def scrape_reddit():
    logging.info("Starting to scrape Reddit data...")
    posts = []
    for post in subreddit.new(limit=100):
        post_data = {
            'id': post.id,
            'title': post.title,
            'body': post.selftext,
            'author': str(post.author) if post.author else 'Unknown'
        }
        # Validate the post_data
        if not post_data['id'] or pd.isna(post_data['id']):
            logging.warning(f"Post with missing or invalid ID detected: {post_data}")
            continue  # Skip posts with missing or invalid IDs
        posts.append(post_data)
    df = pd.DataFrame(posts)
    # Drop rows with NaN IDs before returning
    df = df.dropna(subset=['id'])
    # Ensure that IDs are string type and unique
    df['id'] = df['id'].astype(str)
    logging.info(f"Scraped {len(df)} valid posts from subreddit '{subreddit_name}'")
    return df

def download_existing_data():
    logging.info("Downloading existing data from MongoDB...")
    existing_data = list(posts_collection.find())
    if existing_data:
        logging.info(f"Downloaded {len(existing_data)} existing posts from MongoDB.")
        return pd.DataFrame(existing_data)
    else:
        logging.info("No existing data found in MongoDB.")
        return pd.DataFrame()

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

def update_lsa_model(new_data, updated_data):
    logging.info("Updating LSA model...")
    new_data = pd.DataFrame(new_data)
    updated_data = pd.DataFrame(updated_data)
    # Load the vectorizer and LSA model
    try:
        vectorizer = joblib.load(vectorizer_path)
        svd_model = joblib.load(model_path)
    except Exception as e:
        logging.error(f"Error loading model or vectorizer: {e}")
        return
    
    # Process new data for the LSA model update and coherence score calculation
    new_data["full"] = new_data["title"] + " " + new_data["body"]
    new_documents = new_data['full'].tolist()

    updated_data["full"] = updated_data["title"] + " " + updated_data["body"]
    texts = updated_data['full'].tolist()

    start_time = datetime.now()
    processed_docs = [preprocess_text(doc) for doc in new_documents]
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]  # Filter out empty documents
    if not processed_docs:
        logging.warning("No valid documents after preprocessing new data. Skipping LSA update.")
        return
    try:
        new_tfidf_matrix = vectorizer.transform([' '.join(doc) for doc in processed_docs])
        svd_model.transform(new_tfidf_matrix)
        # Extract topics by finding the top words in each component
        terms = vectorizer.get_feature_names_out()
        new_lsa_topics = []
        for idx, topic in enumerate(svd_model.components_):
            topic_terms = [terms[i] for i in topic.argsort()[:-10 - 1:-1]]
            new_lsa_topics.append(topic_terms)
        logging.info(f"Extracted {len(new_lsa_topics)} topics from the new LSA model.")    
    except Exception as e:
        logging.error(f"Error during LSA model transformation: {e}")
        return
    # Process the updated data for coherence calculation
    processed_data = [preprocess_text(doc) for doc in texts]
    processed_data = [doc for doc in processed_data if len(doc) > 0]  # Filter out empty documents
    
    if not processed_data:
        logging.warning("No valid documents after preprocessing updated data. Skipping coherence score calculation.")
        return

    # Create a dictionary for coherence calculation
    dictionary = Dictionary(processed_data)

    # Calculate coherence score using gensim's CoherenceModel
    try:
        coherence_model = CoherenceModel(topics=new_lsa_topics, texts=processed_data, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        # Save coherence score to MongoDB
        coherence_entry = {
            'coherence_score': coherence_score,
            'timestamp': pd.Timestamp.now(),  # Record the current time
            'latency' : elapsed_seconds
        }
        coherence_collection.insert_one(coherence_entry)
        logging.info(f"Coherence score calculated and saved to MongoDB: {coherence_score}")
    
    except Exception as e:
        logging.error(f"Error calculating coherence score: {e}")

def compare_and_update():
    logging.info("Starting the data processing workflow.")
    
    # Assume scrape_reddit() and download_existing_data() are defined elsewhere
    new_data = scrape_reddit()  # Fetch new data
    existing_data = download_existing_data()  # Fetch existing data

    # Ensure 'id' columns are of type string
    new_data['id'] = new_data['id'].astype(str)
    logging.info("Fetched new data with %d entries.", len(new_data))
    
    if not existing_data.empty:
        existing_data['id'] = existing_data['id'].astype(str)
        existing_data = existing_data.dropna(subset=['id'])
        existing_ids = set(existing_data['id'])
        logging.info("Fetched existing data with %d entries.", len(existing_data))
        
        # Find newly seen data
        newly_seen_data = new_data[~new_data['id'].isin(existing_ids)]
        logging.info("Detected %d new entries.", len(newly_seen_data))
        
        # Combine the existing and new data to create an updated dataset
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['id']).reset_index(drop=True)
    else:
        newly_seen_data = new_data
        updated_data = new_data
        logging.info("No existing data found. Treating all new data as updates.")

    if not newly_seen_data.empty:
        logging.info("Updates detected in the data.")
        posts_collection.delete_many({})
        upload_post(updated_data)
        logging.info("MongoDB updated")
        # Only update the LSA model with the newly seen data
        logging.info("Proceeding to update the LSA model with newly seen data...")
        update_lsa_model(newly_seen_data, updated_data)
        logging.info("LSA model updated with newly seen data")
    else:
        logging.info("No new updates detected in the data.")

    logging.info("Data processing workflow completed.")
    # Update the database with the combined updated dataset


# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2024, 7, 7, 16, 0, 0, tz='Asia/Jakarta'),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'cdc_update_lsa_data',
    default_args=default_args,
    description='A DAG to scrape data from Reddit, compare with existing data in MongoDB, and update if necessary',
    schedule_interval=timedelta(minutes=20),
)

# Define the PythonOperator
scrape_and_model_task = PythonOperator(
    task_id='scrape_and_model_task',
    python_callable=compare_and_update,
    dag=dag,
)

# Add the task to the DAG context
scrape_and_model_task