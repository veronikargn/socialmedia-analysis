import pandas as pd
import logging
from pymongo import MongoClient
import praw
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
from datetime import datetime
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import pendulum
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta

# Download NLTK stopwords
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

client = MongoClient('MONGODB_CONNECTIO') 
db = client['batch_lda']
posts_collection = db['posts']
coherence_collection = db['coherence_scores']

model_directory = '/opt/airflow/lda_model'
model_path = os.path.join(model_directory, 'lda_model')

def upload_post(posts):
    valid_data = posts.dropna(subset=['id']).to_dict('records')
    for doc in valid_data:
        # Ensure _id is set correctly for MongoDB uniqueness
        doc['_id'] = doc['id']  
        # Insert updated data with unique IDs
    if valid_data:  # Ensure we only insert if there is valid data
        posts_collection.insert_many(valid_data)
        logging.info(f"Data updated successfully in MongoDB! Inserted {len(valid_data)} records.")
    else:
        logging.info("No valid data to insert after merging.")
    
def scrape_reddit():
    logging.info("Starting to scrape Reddit data...")
    posts = []
    for post in subreddit.new(limit=100):  # Adjusted the limit as per requirement
        post_data = {
            'id': post.id,
            'title': post.title,
            'body': post.selftext,
            'author': str(post.author) if post.author else 'Unknown'  # Handle case where author might be None
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

def saving(coherence_lda, lda_model, elapsed_time):
    # coherence
    if np.isnan(coherence_lda):
        logging.info("Calculated coherence score is NaN. Skipping recording.")
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

def update_lda_model(data):
    logging.info(f"Downloaded {len(data)} existing posts for preprocessing.")
    data["full"] = data["title"] + " " + data["body"]  
    texts = data['full'].tolist()
    lda_model = LdaModel.load(model_path)
    logging.info("Loaded existing LDA model.")
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
    corpus = [dictionary.doc2bow(doc) for doc in filtered_docs]

    lda_model.update(corpus, passes=10, iterations=50)
    lda_model.sync_state()
    logging.info("Updated existing LDA model with new data.")

    coherence_model_lda = CoherenceModel(model=lda_model, texts=filtered_docs, dictionary=lda_model.id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logging.info(f"Coherence score recorded: {coherence_lda}")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    saving(coherence_lda, lda_model, elapsed_time.total_seconds())

def compare_and_update_data():
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
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=['id']).reset_index(drop=True)
        logging.info("Detected %d new entries.", len(newly_seen_data))
    else:
        newly_seen_data = new_data
        updated_data = new_data
        logging.info("No existing data found. Treating all new data as updates.")

    if not newly_seen_data.empty:
        logging.info("Updates detected in the data.")
        posts_collection.delete_many({})
        upload_post(updated_data)
        logging.info("MongoDB updated")
        logging.info("Proceeding to update the LDA model...")
        update_lda_model(updated_data)
        logging.info("LDA model updated")

    else:
        logging.info("No updates detected in the data.")
    logging.info("Data processing workflow completed.")

# Define the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2024, 7, 8, 7, 0, 0, tz='Asia/Jakarta'),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'cdc_update_lda_data',
    default_args=default_args,
    description='A DAG to scrape data from Reddit, compare with existing data in MongoDB, and update if necessary',
    schedule_interval=timedelta(minutes=20),
)

# Define the PythonOperator
scrape_and_model_task = PythonOperator(
    task_id='scrape_and_model_task',
    python_callable=compare_and_update_data,
    dag=dag,
)

# Add the task to the DAG context
scrape_and_model_task