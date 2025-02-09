"""
Script loads the latest trained model, data for inference and predicts results.
Imports necessary packages and modules.
"""
import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from scipy.sparse import load_npz

# Comment this line if you have problems with MLFlow installation
import mlflow
mlflow.pytorch.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

from utils import get_project_dir, configure_logger
# Change to CONF_FILE = "settings.json" if you have problems with env variables
#CONF_FILE = os.getenv('CONF_PATH')
CONF_FILE = get_project_dir("settings.json")

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initializes parser for command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file", 
                    help="Specify inference data file", 
                    default=conf['inference']['inp_table_name'])
parser.add_argument("--out_path", 
                    help="Specify the path to the output table")

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

def get_latest_model_path() -> str:
    """Gets the path of the latest saved model"""
    latest = None
    for (dirpath, dirnames, filenames) in os.walk(MODEL_DIR):
        for filename in filenames:
            if not latest or datetime.strptime(latest, conf['general']['datetime_format'] + '_best_model.pth') < \
                    datetime.strptime(filename, conf['general']['datetime_format'] + '_best_model.pth'):
                latest = filename
    return os.path.join(MODEL_DIR, latest)


def get_model_by_path(path: str) -> TextClassifier:
    """Loads and returns the specified model"""
    try:
        model = TextClassifier(148909)
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f'Model loaded from {path}')
        return model
    except Exception as e:
        logger.error(f'An error occurred while loading the model: {e}')
        sys.exit(1)

def collate_fn( batch):
    # Unzip the batch
    inputs, labels = zip(*batch)

    inputs = torch.stack(inputs, dim=0)

    # Convert labels to a tensor - you might need to modify this depending on your label format
    labels = torch.stack([torch.tensor(label) for label in labels])

    return inputs, labels


def get_inference_data(path: str, batch_size: int = 64) -> DataLoader:
    """loads and returns data for inference """

    test_tfidf_matrix_path = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


    X_test_tfidf = load_npz(test_tfidf_matrix_path)


    test_sentiment_path = os.path.join(DATA_DIR, conf['inference']['inf_y_table_name'])


    test_sentiment = np.load(test_sentiment_path)



    X_test_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(X_test_tfidf.nonzero()),
        torch.FloatTensor(X_test_tfidf.data),
        torch.Size(X_test_tfidf.shape)
    )

    y_test_tensor = torch.LongTensor(test_sentiment)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return test_loader

def predict_results(model: TextClassifier, criterion: nn.BCELoss(), val_loader: DataLoader) -> pd.DataFrame:
    """Predict de results and join it with the infer_data"""

    # Validation loop
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    all_val_labels = []
    all_val_predictions = []

    with torch.no_grad():

        for batch in val_loader:
            #print(len(batch))  # Debug: Check the batch structure
            if len(batch) == 2: #isinstance(batch, tuple) and
                data, target = batch
                labels = target.float().view(-1) #.unsqueeze(1)
            else:
                data = batch[0]
                labels = None

            output = model(data)
            output = output.view(-1)
            if labels is not None:

                loss = criterion(output, labels)
                val_loss += loss.item()
                predicted = (output.data > 0.5).float()  # Threshold prediction
                total += target.size(0)
                correct += (predicted == labels).sum().item()


                all_val_labels.append(labels.cpu().numpy())
            else:
                predicted = (output.data > 0.5).float() # Get indices of max log-probability

            all_val_predictions.append(predicted.cpu().numpy())
            #print(all_val_predictions)

        # Calculate accuracy if targets are available
        if total > 0:
            val_accuracy = 100 * correct / total
            logger.info(f'Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%')
        else:
            logger.info('No target labels available for accuracy computation.')

        # Prepare results DataFrame
        #print(all_val_labels)
        if all_val_labels:
            results_df = pd.DataFrame({
                'Actual': np.concatenate(all_val_labels),
                'Predicted': np.concatenate(all_val_predictions)
            })
        else:
            results_df = pd.DataFrame({
                'Predicted': np.concatenate(all_val_predictions)
            })

        return results_df


def store_results(results: pd.DataFrame, path: str = None) -> None:
    """Store the prediction results in 'results' directory with current datetime as a filename"""
    if not path:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)
    pd.DataFrame(results).to_csv(path, index=False)
    logger.info(f'Results saved to {path}')


def main():
    """Main function"""
    logger.add("_logs_for_test.txt", level="INFO")

    # Log a test message to ensure logger is working
    logger.info("logger setup complete. Starting testing...")


    args = parser.parse_args()

    model = get_model_by_path(get_latest_model_path())
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, infer_file))
    results = predict_results(model,  nn.BCELoss(), infer_data)
    store_results(results, args.out_path)

    logger.info(f'Prediction results: {results}')


if __name__ == "__main__":
    main()