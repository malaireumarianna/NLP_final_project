# Importing required libraries
import numpy as np

from scipy.sparse import save_npz
import os
import sys
import json
import pandas as pd
import torch
import re
#from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Needed for the WordNet Lemmatizer
nltk.download('omw-1.4')  # Optionally, for additional WordNet multilingual support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# Download the stopwords dataset
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
vectorizer_tfidf = TfidfVectorizer()
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, '../data'))
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = get_project_dir("settings.json") #os.getenv('CONF_PATH')

# Load configuration settings from JSON

with open(CONF_FILE, "r") as file:
    conf = json.load(file)


DATA_DIR = get_project_dir(conf['general']['data_dir'])

class DataProcessor:
    def __init__(self, data_dir, train_file, test_file):
        self.data_dir = data_dir
        self.train_file = train_file
        self.test_file = test_file

        self.lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
        self.vectorizer_tfidf = TfidfVectorizer()  # Assuming TfidfVectorizer is imported
        self.nltk_stopwords = set(stopwords.words('english'))

    def load_data(self, file_path):
        return pd.read_csv(os.path.join(self.data_dir, file_path))


    def clean_text(self, text):
        text = text.lower()  # convert to lowercase to maintain consistency
        text = re.sub(r'\s+', ' ', text)  # replace multiple whitespaces with single space
        text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        return text


    #tokenize text using NLTK
    def nltk_tokenize(self, text):
        # Tokenizes a string into words
        tokens = word_tokenize(text)
        return tokens

    def remove_stopwords(self, tokens):
        # Filter tokens to remove stopwords
        return [token for token in tokens if token.lower() not in self.nltk_stopwords]

        #return [token for token in tokens if token not in STOP_WORDS]

    def nltk_lemmatize(self, tokens):
        # Lemmatize each token in the list of tokens
        return [lemmatizer.lemmatize(token) for token in tokens]

    def collate_fn(self, batch):
        # Unzip the batch
        inputs, labels = zip(*batch)

        # Since inputs are already sparse tensors and don't need batching
        # Convert them to a stacked sparse tensor if you are processing multiple inputs at once
        inputs = torch.stack(inputs, dim=0)

        # Convert labels to a tensor - you might need to modify this depending on your label format
        labels = torch.stack([torch.tensor(label) for label in labels])

        return inputs, labels



    def data_preprocess(self):

        train = self.load_data(self.train_file)
        test = self.load_data(self.test_file)

        train['review'] = train['review'].apply(self.clean_text)
        test['review'] = test['review'].apply(self.clean_text)

        train['tokens'] = train['review'].apply(self.nltk_tokenize)
        test['tokens'] = test['review'].apply(self.nltk_tokenize)

        train['tokens'] = train['tokens'].apply(self.remove_stopwords)
        test['tokens'] = test['tokens'].apply(self.remove_stopwords)

        train['lemmas'] = train['tokens'].apply(self.nltk_lemmatize)
        test['lemmas'] = test['tokens'].apply(self.nltk_lemmatize)

        # Convert lists of tokens/lemmas back to a single string per document
        train['lemmas_joined'] = train['lemmas'].apply(lambda x: ' '.join(x))
        test['lemmas_joined'] = test['lemmas'].apply(lambda x: ' '.join(x))


        # Fit the vectorizer on the training data and transform both train and test data
        X_train_tfidf = vectorizer_tfidf.fit_transform(train['lemmas_joined'])
        X_test_tfidf = vectorizer_tfidf.transform(test['lemmas_joined'])

        # Map 'positive' to 1 and 'negative' to 0
        label_mapping = {'positive': 1, 'negative': 0}
        train['sentiment'] = train['sentiment'].map(label_mapping)
        test['sentiment'] = test['sentiment'].map(label_mapping)

        save_npz(os.path.join(DATA_DIR, conf['train']['table_name']), X_train_tfidf)
        save_npz(os.path.join(DATA_DIR, conf['inference']['inp_table_name']), X_test_tfidf)

        np.save(os.path.join(DATA_DIR, conf['train']['y_table_name']), train['sentiment'].values)
        np.save(os.path.join(DATA_DIR, conf['inference']['inf_y_table_name']), test['sentiment'].values)

        print(f"Training data saved")

        print(f"Inference data saved")


# Main execution
if __name__ == "__main__":
    DATA_DIR = get_project_dir(conf['general']['data_dir'])
    train_file = 'final_project_train_dataset/train.csv'
    test_file = 'final_project_test_dataset/test.csv'

    #os.path.join(self.data_dir, file_path)
    data_proc = DataProcessor(DATA_DIR, train_file, test_file)
    data_proc.data_preprocess()
