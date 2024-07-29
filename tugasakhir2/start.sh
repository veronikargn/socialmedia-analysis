#!/bin/bash
if [ "$SCRIPT" == "streamer" ]; then
  python reddit_mongo.py
elif [ "$SCRIPT" == "lsa_updater" ]; then
  python processing.py
else
  echo "Invalid SCRIPT variable"
fi