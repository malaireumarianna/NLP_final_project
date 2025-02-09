
## Project structure:

This project has a modular structure, where each folder has a specific duty.

```
NLP_final_project
├── data                      # Data files used for training and inference (it can be generated with data_generation.py script)
│   ├── test.csv
│   └── train.csv
├── data_process              # Scripts used for data processing and generation
│   ├── data_preprocess.py
│   ├── data_download.py
│   └── __init__.py           
├── inference                 # Scripts and Dockerfiles used for inference
│   ├── Dockerfile
│   ├── run.py
│   └── __init__.py
├── models                    # Folder where trained models are stored
│   └── various model files
├── training                  # Scripts and Dockerfiles used for training
│   ├── Dockerfile
│   ├── train.py
│   └── __init__.py
├── notebooks                  # Scripts used for analysis of dataset
│   └── analysis.ipyb
├── utils.py                  # Utility functions and classes that are used in scripts
├── settings.json             # All configurable parameters and settings
└── README.md
```

## Settings:
The configurations for the project are managed using the `settings.json` file. It stores important variables that control the behaviour of the project. Examples could be the path to certain resource files, constant values, hyperparameters for an ML model, or specific settings for different environments. Before running the project, ensure that all the paths and parameters in `settings.json` are correctly defined.
Keep in mind that you may need to pass the path to your config to the scripts. For this, you may create a .env file or manually initialize an environment variable as `CONF_PATH=settings.json`.
Please note, some IDEs, including VSCode, may have problems detecting environment variables defined in the .env file. This is usually due to the extension handling the .env file. If you're having problems, try to run your scripts in a debug mode, or, as a workaround, you can hardcode necessary parameters directly into your scripts. Make sure not to expose sensitive data if your code is going to be shared or public. In such cases, consider using secret management tools provided by your environment.

## Data:
This is a dataset for binary sentiment classification. This is a set of 50,000 polar movie reviews for training and testing: 

train.csv 
test.csv

Data files should be downloaded from Epam platform and stored in the `data` directory, as train set is too big for storing on GitHub.
This is done in `data_process/data_donwload.py`, so start from running this script.

For generating correctly preprocessed data, use the script located at `data_process/data_preprocess.py`. The generated data is used to train the model and to test the inference. 
Script is doing simple cleaning of text, removes stop words, tokenize and Lemmatize. Then, input features are  saved in .npz files, and target labels into .npy files.
So, as result of script execution, `data` directory should contain these files:

test.csv
test.npz
test_sentiment.npy
train.csv
train.npz
train_sentiment.npy


## Training:
The training phase of the ML pipeline includes the actual training of the NN model, and the evaluation on validation set created from training data. All of these steps are performed by the script `training/train.py`.

1. To train the model using Docker: 

- Build the training Docker image. If the built is successfully done, it will automatically train the model:
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t training_image .
```
- You may run the container with the following parameters to ensure that the trained model is here:
```bash
docker run -it training_image /bin/bash
```
In this directory can be found file `_logs_for_train.txt` which contains detailed logs for training process, accuracy and loss for testing the model on subset of train data.

To read this file in terminal, use this command:
```bash
cat _logs_for_train.txt
```

- Enter the repository with models and copy the name of saved model by using commands below:
```bash
cd models
```
```bash
ls
```
Here you will see model, so copy it's name for future.

- Then exit to be able to run copying of model by docker command:
```bash
exit
```
Then, move the trained model from the directory inside the Docker container `/app/models` to the local machine using:

```bash
docker cp <container_id>:/app/models/<model_name>.pth ./models
```
Replace `<container_id>` with your running Docker container ID and `<model_name>.pth` with your model's name.

1. Alternatively, the `train.py` script can also be run locally as follows:

```bash
python3 training/train.py
```

## Inference:
Once a model has been trained, it can be used to make predictions on new data in the inference stage. The inference stage is implemented in `inference/run.py`.

1. To run the inference using Docker, use the following commands:

- Build the inference Docker image:
```bash
docker build -f ./inference/Dockerfile --build-arg model_name=<model_name>.pth --build-arg settings_name=settings.json -t inference_image .
```
- Run the inference Docker container:
```bash
docker run -v /path_to_your_local_model_directory:/app/models -v /path_to_your_input_folder:/app/input -v /path_to_your_output_folder:/app/output inference_image
```
- Or you may run it with the attached terminal using the following command:
```bash
docker run -it inference_image /bin/bash  
```
In this directory can be found file `_logs_for_test.txt` which contains detailed logs for inference.
To read this file in terminal, use this command:
```bash
cat _logs_for_test.txt
```

After that ensure that you have your results in the `results` directory in your inference container. It should contain csv file with the Predicted values.


2. Alternatively, you can also run the inference script locally:

```bash
python inference/run.py
```

Replace `/path_to_your_local_model_directory`, `/path_to_your_input_folder`, and `/path_to_your_output_folder` with actual paths on your local machine or network where your models, input, and output are stored.



