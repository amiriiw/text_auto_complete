"""-------------------------------------------------------------------------
Welcome, this is amiriiw. This is a simple project about text auto-complete.
In this file, we will use the models to complete texts.
----------------------------------------------------"""
import pickle  # https://docs.python.org/3/library/pickle.html
import sqlite3  # https://docs.python.org/3/library/sqlite3.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from tensorflow.keras.preprocessing.sequence import pad_sequences  # https://www.tensorflow.org/guide/keras
"""-----------------------------------------------------------------------------------------------------"""


class TextAutoComplete:
    def __init__(self, model_path, db_path='autocomplete.db'):
        self._model_path = model_path
        self._db_path = db_path
        self._tokenizer = self._load_tokenizer()
        self._model = self._load_model()
        self._conn = sqlite3.connect(self._db_path)
        self._cursor = self._conn.cursor()
        self._create_table()

    @staticmethod
    def _load_tokenizer():
        with open('tokenizer.pickle', 'rb') as handle:
            return pickle.load(handle)

    def _load_model(self):
        return tf.keras.models.load_model(self._model_path)

    def _create_table(self):
        self._cursor.execute('''CREATE TABLE IF NOT EXISTS autocomplete_data
                                (user_text TEXT, added_words TEXT, final_text TEXT)''')

    def autocomplete_text(self, input_text, num_words=2):
        token_list = self._tokenizer.texts_to_sequences([input_text])[0]
        max_sequence_len = self._model.input_shape[1]
        for _ in range(num_words):
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
            predicted_probs = self._model.predict(token_list, verbose=0)
            predicted_word_index = np.argmax(predicted_probs, axis=-1)[0]
            predicted_word = self._tokenizer.index_word[predicted_word_index]
            input_text += " " + predicted_word
            token_list = np.append(token_list, predicted_word_index)
        return input_text.split()[-num_words:], input_text

    def save_to_db(self, user_text, new_words, completed_text):
        self._cursor.execute(
            "INSERT INTO autocomplete_data (user_text, added_words, final_text) VALUES (?, ?, ?)",
            (user_text, new_words, completed_text)
        )
        self._conn.commit()

    def close_connection(self):
        self._conn.close()

    def run(self, text_input):
        words_added, text_completed = self.autocomplete_text(text_input)
        self.save_to_db(text_input, ' '.join(words_added), text_completed)
        self.close_connection()
        return text_input, words_added, text_completed


if __name__ == "__main__":
    model_file_path = 'Text_auto_complete.h5'
    user_input_text = input("Enter a text: ")
    auto_complete = TextAutoComplete(model_file_path)
    original_user_text, added_words, final_text = auto_complete.run(user_input_text)
    print("Original Text:", original_user_text)
    print("Added Words:", ' '.join(added_words))
    print("Final Text:", final_text)
"""------------------------------"""
