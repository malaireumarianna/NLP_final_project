
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



## Dataset description

![image](https://github.com/user-attachments/assets/7a82cd69-c3a6-418a-b75b-1d31a8664aee)

```bash
Top 10 positive words: [('br', 43805), ('film', 31762), ('movie', 28754), ('one', 20914), ('like', 13538), ('good', 11520), ('great', 10118), ('story', 9863), ('see', 9523), ('time', 9409)]
Top 10 negative words: [('br', 46995), ('movie', 38097), ('film', 28097), ('one', 19987), ('like', 17494), ('even', 12062), ('bad', 11389), ('good', 11370), ('would', 10988), ('really', 9857)]
```
The most popular words describe used in dataset show its relation to movie reviews, in most popular positive words we can see some positive feedback as "like", "great". In negative there are some words showing negative feedback as "bad".

![image](https://github.com/user-attachments/assets/48a6bd13-841c-426c-bffe-39ea58f08ba6)

Both positive and negative sentiments have a median polarity near zero but the positive sentiment category shows a median slightly above zero.
Both categories show a wide range of polarity from -1 to 1, indicating variability in the intensity of sentiment within each category. This suggests that there are positive texts with negative polarity scores and vice versa, which could be due to sarcasm, mixed sentiments, or inaccuracies in the polarity scoring method.
There are numerous outliers in both categories, especially in the positive category, where some reviews have extremely negative polarity scores

![image](https://github.com/user-attachments/assets/9f8e99a2-bc2c-4990-8045-677d2abdc47c)

Both positive and negative reviews show a similar distribution shape, with most reviews being relatively short. The distribution is heavily skewed to the right, indicating that longer reviews are less common.
The two distributions largely overlap, but the positive sentiment reviews appear slightly more concentrated in the lower word count range than negative reviews.

![image](https://github.com/user-attachments/assets/b485ebca-f8c1-4d48-9147-1db7f1ec1c13)

The most prominent words in the cloud are "movie," "time," "good," "great," and "story."
The usage of words like "better," "best," "original," and "perfect" show that reviewers often compare the movie to others.

![image](https://github.com/user-attachments/assets/72c1f523-2fec-406b-8124-fb9da4e46bd2)


## Data processing and model selection:

I compared `WordNet Lemmatizer` and `PorterStemmer`. As well as two vectorizer methods - `CountVectorizer()` and `TfidfVectorizer()`.
With NaiveBayes Classifier  `WordNet Lemmatizer` together with `TfidfVectorizer()` works slightly better on our dataset with accuracy 86.53%, so it would be the choice for the next models. 
The next model used was Random Forest classifiers, which performed slightly worse than NB with accuracy 85.46%.
The last model use was custom created Neural Network for binary classification:
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
        x = self.sigmoid(x)  # Ensure output is between 0 and 1
        return x
Its accuracy is: 88.68%.
The best-performing model was used for further Docker containerization.


