#!/bin/bash
if [ "$SCRIPT" == "streamer" ]; then
  python reddit_to_mongo.py
elif [ "$SCRIPT" == "lda_updater" ]; then
  python update_lda.py
else
  echo "Invalid SCRIPT variable"
fi