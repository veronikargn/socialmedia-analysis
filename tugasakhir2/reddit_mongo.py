import praw
from pymongo import MongoClient
import pandas as pd
import json
import logging
from kafka import KafkaProducer
from datetime import datetime

# MongoDB connection setup
client = MongoClient('MONGODBCONNECTION')
db = client['realtime_lsa']
collection = db['posts']

# Connect to Kafka
KAFKA_TOPIC = 'reddit_updates'
producer = KafkaProducer(bootstrap_servers='kafka:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# PRAW (Python Reddit API Wrapper) setup
reddit = praw.Reddit(
    client_id="CLIENT_ID",
    client_secret="CLIENT_SECRET",
    user_agent="USER_AGENT",
)

# Choose the subreddit to stream from
subreddit = reddit.subreddit('beauty')

def download_existing_data():
    logging.info("Downloading existing data from MongoDB...")
    existing_data = list(collection.find())
    if existing_data:
        logging.info(f"Downloaded {len(existing_data)} existing posts from MongoDB.")
        return pd.DataFrame(existing_data)
    else:
        logging.info("No existing data found in MongoDB.")
        return pd.DataFrame()

def compare_mongo(document):
    existing_data = download_existing_data()
    
    if not existing_data.empty:
        existing_data['id'] = existing_data['id'].astype(str)
        existing_data = existing_data.dropna(subset=['id'])
        existing_ids = set(existing_data['id'])
        logging.info("Fetched existing data with %d entries.", len(existing_data))
        
        # Check if the document's id is in the existing ids
        if str(document['id']) in existing_ids:
            logging.info("Document with id %s already exists. No update needed.", document['id'])
        else:
            # save to mongo
            print("data is new! sending message to processor")
            collection.insert_one(document)
            logging.info("Inserted new document with id %s.", document['id'])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Send post title and body to Kafka
            dummy_message = {"event": "new_post", "title": document['title'], "body": document['body'], "time": str(timestamp)}
            producer.send(KAFKA_TOPIC, dummy_message)
            logging.info(f"Sent message to Kafka: {dummy_message}")
    else:
        # No existing data, insert the new document
        collection.insert_one(document)
        logging.info("No existing data found. Inserted new document with id %s.", document['id'])

# Function to stream data and store it in MongoDB
for submission in subreddit.stream.submissions():
    # Create a document with the relevant fields
    post = {
        'id': submission.id,
        'title': submission.title,
        'body': submission.selftext,
        'author': str(submission.author) if submission.author else 'Unknown'
    }
    print("streamed new data")
    compare_mongo(post)
    # Insert the document into MongoDB