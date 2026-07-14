import os
import sys

# Add the current directory to sys.path so we can import chatbot
sys.path.append(os.getcwd())

from chatbot.data_loader import built_database

print("Starting database rebuild...")
built_database()
print("Database rebuild complete.")
