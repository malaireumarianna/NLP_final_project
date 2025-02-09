import argparse
import os
import sys
import json
from scipy.sparse import load_npz
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from loguru import logger
import nltk
from typing import Tuple
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')  # Needed for the WordNet Lemmatizer
nltk.download('omw-1.4')  # Optionally, for additional WordNet multilingual support
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
lemmatizer = WordNetLemmatizer()
vectorizer_tfidf = TfidfVectorizer()
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# Comment this line if you have problems with MLFlow installation
import mlflow
mlflow.pytorch.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logger

# Loads configuration settings from JSON
#CONF_FILE = os.getenv('CONF_PATH', os.path.join(ROOT_DIR, 'settings.json'))


CONF_FILE = get_project_dir("settings.json") #os.getenv('CONF_PATH')
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify inference data file",
                    default=conf['train']['table_name'])
parser.add_argument("--model_path",
                    help="Specify the path for the output model")

class DataProcessor:
    def __init__(self) -> None:
        pass

    def collate_fn(self, batch):
        # Unzip the batch
        inputs, labels = zip(*batch)

        # Since inputs are already sparse tensors and don't need batching
        # Convert them to a stacked sparse tensor if you are processing multiple inputs at once
        inputs = torch.stack(inputs, dim=0)

        # Convert labels to a tensor - you might need to modify this depending on your label format
        labels = torch.stack([torch.tensor(label) for label in labels])

        return inputs, labels



    def data_prep(self) -> Tuple[int, DataLoader, DataLoader]:
        # Load the sparse matrices
        train_tfidf_matrix_path = os.path.join(DATA_DIR, conf['train']['table_name'])
        test_tfidf_matrix_path = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

        X_train_tfidf = load_npz(train_tfidf_matrix_path)
        X_test_tfidf = load_npz(test_tfidf_matrix_path)

        # Load the numpy arrays
        train_sentiment_path = os.path.join(DATA_DIR, conf['train']['y_table_name'])
        test_sentiment_path = os.path.join(DATA_DIR, conf['inference']['inf_y_table_name'])

        train_sentiment = np.load(train_sentiment_path)
        test_sentiment = np.load(test_sentiment_path)


        # Convert sparse matrices to PyTorch sparse tensors
        X_train_tensor = torch.sparse_coo_tensor(
            torch.LongTensor(X_train_tfidf.nonzero()),
            torch.FloatTensor(X_train_tfidf.data),
            torch.Size(X_train_tfidf.shape)
        )
        num_features = X_train_tensor.shape[1]


        X_test_tensor = torch.sparse_coo_tensor(
            torch.LongTensor(X_test_tfidf.nonzero()),
            torch.FloatTensor(X_test_tfidf.data),
            torch.Size(X_test_tfidf.shape)
        )


        # Convert labels to integer tensors
        y_train_tensor = torch.LongTensor(train_sentiment)
        y_test_tensor = torch.LongTensor(test_sentiment)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        validation_split = 0.2  # e.g., 20% of the data goes to the validation set
        num_train = len(train_dataset)
        num_val = int(num_train * validation_split)
        num_train = num_train - num_val

        # Split the dataset
        train_subset, val_subset = random_split(train_dataset, [num_train, num_val])

        # Create data loaders
        batch_size = 64
        # Create DataLoaders for each subset
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn)

        return num_features, train_loader,  val_loader


class TextClassifier(nn.Module):
    def __init__(self, num_features):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # output is between 0 and 1
        return x


class Training:
    def __init__(self, model):
        self.model = model
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.best_val_loss = float('inf')  # Initialize best validation loss to a very large value
        self.best_model_path = os.path.join(MODEL_DIR, datetime.now().strftime(
            conf['general']['datetime_format']) + '_best_model.pth')

    def run_training(self, train_loader: DataLoader, val_loader: DataLoader):


        # Training loop
        for epoch in range(3):  # Number of epochs
            running_loss = 0.0  # Initialize the running loss at the start of the epoch
            total_instances = 0
            self.model.train()

            for data, target in train_loader:
                labels = target.float().unsqueeze(1)  # BCE requires float target tensor

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)  # Ensure outputs are [0, 1] via sigmoid in model

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                total_instances += labels.size(0)

            # Log training loss for each epoch
            average_loss = running_loss / total_instances if total_instances > 0 else 0
            logger.info(f'Epoch {epoch + 1}, Training Loss: {average_loss}')

            # Validation loop
            self.model.eval()
            correct = 0
            total = 0
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    labels = target.float().unsqueeze(1)  # BCE requires float target tensor

                    output = self.model(data)
                    loss = self.criterion(output, labels)
                    val_loss += loss.item()

                    predicted = (output.data > 0.5).float()  # Threshold prediction
                    total += target.size(0)
                    correct += (predicted == target.unsqueeze(1)).sum().item()

            val_accuracy = 100 * correct / total
            logger.info(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}, Accuracy: {val_accuracy}%')

        self.save()  # Save the best model

    def save(self,) -> None:
        logger.info("Saving the model...")

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        torch.save(self.model.state_dict(), self.best_model_path)

def main():
    logger.add("_logs_for_train.txt", level="INFO")

    
    logger.info("Logging setup complete. Starting training...")


    data_proc = DataProcessor()

    num_features, train_loader, valid_loader = data_proc.data_prep()
    model = TextClassifier(num_features)
    tr = Training(model)
    tr.run_training( train_loader, valid_loader)
    #tr.save(datetime.now().strftime(conf['general']['datetime_format']) + '_model.pth')

if __name__ == "__main__":
    main()
