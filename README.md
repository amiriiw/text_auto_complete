# Text Auto-Complete Project

Welcome to the **Text Auto-Complete Project**! This project is designed to train a model for completing text inputs using a Persian dataset, and then use that model to suggest text completions in real-time and store the results in an SQLite database.

## Overview

This project consists of two main components:

1. **text_auto_complete_model_trainer.py**: This script is responsible for training a model to autocomplete text using a Persian dataset.
2. **text_auto_complete.py**: This script uses the trained model to suggest text completions and stores the results in an SQLite database.

## Libraries Used

The following libraries are used in this project:

- **[joblib](https://joblib.readthedocs.io/en/stable/)**: Used for saving and loading the model.
- **[pickle](https://docs.python.org/3/library/pickle.html)**: Used for saving and loading the tokenizer.
- **[numpy](https://numpy.org/devdocs/user/absolute_beginners.html)**: Used for numerical operations and data manipulation.
- **[pandas](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)**: Used for handling the dataset and loading the data.
- **[tensorflow](https://www.tensorflow.org/)**: Used for model building, training, and prediction.
- **[sqlite3](https://docs.python.org/3/library/sqlite3.html)**: Used for creating and interacting with a local SQLite database to store autocomplete results.

## Detailed Explanation

### `text_auto_complete_model_trainer.py`

This script is the core of the project, responsible for training the text autocomplete model. The key components of the script are:

- **TrainModel Class**: This class handles the entire process from loading the dataset to training and saving the model. The main methods include:
  - `preprocess_data()`: Cleans the text data by removing unwanted characters.
  - `tokenize_text()`: Tokenizes the text data and saves the tokenizer for later use.
  - `prepare_sequences()`: Prepares input sequences for training the model by creating n-gram sequences and padding them.
  - `build_model()`: Constructs a Sequential model using Embedding, Bidirectional LSTM, and Dense layers.
  - `train_model()`: Trains the model on the prepared sequences and saves it for later use.
  - `run()`: Executes the entire training process in sequence.

### `text_auto_complete.py`

This script uses the trained model to autocomplete text inputs and stores the results in an SQLite database. The key components of the script are:

- **TextAutoComplete Class**: This class handles the model loading, text autocompletion, and database interactions. The main methods include:
  - `_load_tokenizer()`: Loads the tokenizer from the saved file.
  - `_load_model()`: Loads the trained model from the saved file.
  - `_create_table()`: Creates an SQLite table (if it doesn't already exist) to store the original text, added words, and the final completed text.
  - `autocomplete_text()`: Predicts the next words in the text based on the model's output.
  - `save_to_db()`: Saves the original text, added words, and final completed text to the SQLite database.
  - `close_connection()`: Closes the connection to the SQLite database.
  - `run()`: Handles the entire process of autocompleting the text and saving the result.

### How It Works

1. **Model Training**:
    - The `text_auto_complete_model_trainer.py` script reads a CSV file containing Persian text data.
    - The text is tokenized, and input sequences are generated for training.
    - A model is trained using the tokenized data and saved for later use.

2. **Text Auto-Completion**:
    - The `text_auto_complete.py` script loads the trained model and tokenizer.
    - The user inputs text, which is tokenized and passed through the model.
    - The model predicts the next words, and the results are saved in the SQLite database.

### Dataset

The dataset used for training the model can be accessed via this [Dataset](https://drive.google.com/drive/folders/1Exnb5z7qXkU2l0x2cbQC_CeJvYT9Cn3d?usp=sharing)

## Installation and Setup

To use this project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/amiriiw/text_auto_complete
    cd text_auto_complete
    ```

2. Install the required libraries:

    ```bash
    pip install joblib numpy pandas tensorflow
    ```

3. Prepare your dataset (a CSV file with a column 'Text' containing Persian text).

4. Run the model training script:

    ```bash
    python text_auto_complete_model_trainer.py
    ```

5. Use the trained model for text auto-completion:

    ```bash
    python text_auto_complete.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
