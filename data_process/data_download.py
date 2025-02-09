import requests
import zipfile
import os
import os
import sys
import json
import pandas as pd
import torch
import re

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

DATA_DIR = get_project_dir(conf['general']['data_dir'])
train_file = 'train.csv'
test_file = 'test.csv'


def download_and_extract_zip(url, extract_to=DATA_DIR ):
    """
    Download a ZIP file from a URL and extract its contents into a directory.

    :param url: The URL to download the ZIP file from.
    :param extract_to: The directory to extract the contents to.
    """
    # Create the directory if it does not exist
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        zip_path = os.path.join(extract_to, 'downloaded_data.zip')
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Zip file downloaded successfully.")

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Files extracted successfully to {extract_to}")

        # Optionally, remove the zip file after extraction
        os.remove(zip_path)
        print("Downloaded zip file removed.")
    else:
        print("Failed to download the file. Status code:", response.status_code)


# Example usage
if __name__ == '__main__':
    download_url = 'https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip'
    download_and_extract_zip(download_url)

    download_url = 'https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_test_dataset.zip'
    download_and_extract_zip(download_url)