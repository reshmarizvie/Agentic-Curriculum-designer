from transformers import pipeline

print("Downloading summarizer...")
pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

print("Downloading NER model...")
pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="simple")

print("✅ NLP models cached locally")
