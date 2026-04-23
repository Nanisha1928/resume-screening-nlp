import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+', '', text)   # remove URLs
    text = re.sub(r'\d+', '', text)       # remove numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)