"""---------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about text auto complete.
    in this file we will train the persian dataset to use as auto complete model.
------------------------------------------------------------------------------"""
import joblib  # https://joblib.readthedocs.io/en/stable/
import pickle  # https://docs.python.org/3/library/pickle.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import pandas as pd  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
import tensorflow as tf  # https://www.tensorflow.org/
from tensorflow.keras.optimizers import Adam  # https://www.tensorflow.org/guide/keras
from tensorflow.keras.models import Sequential  # https://www.tensorflow.org/guide/keras
from tensorflow.keras.preprocessing.text import Tokenizer  # https://www.tensorflow.org/guide/keras
from tensorflow.keras.preprocessing.sequence import pad_sequences  # https://www.tensorflow.org/guide/keras
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional  # https://www.tensorflow.org/guide/keras
"""-------------------------------------------------------------------------------------------------------------"""


class TrainModel:
    def __init__(self, dataset_path, model_name):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.dataset = pd.read_csv(self.dataset_path)
        self.tokenizer = None
        self.total_words = 0
        self.max_sequence_len = 0
        self.model = None

    def preprocess_data(self):
        print(f"Records: {self.dataset.shape[0]}")
        print(f"Fields: {self.dataset.shape[1]}")
        self.dataset['Text'] = self.dataset['Text'].str.replace(u'\xa0', u' ')
        self.dataset['Text'] = self.dataset['Text'].str.replace('\u200a', ' ')

    def tokenize_text(self):
        self.tokenizer = Tokenizer(oov_token='<oov>')
        self.tokenizer.fit_on_texts(self.dataset['Text'])
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.total_words = len(self.tokenizer.word_index) + 1

    def prepare_sequences(self):
        input_sequences = []
        for line in self.dataset['Text']:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        print(f"Total input sequences: {len(input_sequences)}")
        self.max_sequence_len = max(len(x) for x in input_sequences)
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre'))
        xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=self.total_words)
        return xs, ys

    def build_model(self):
        self.model = Sequential([
            Embedding(self.total_words, 100, input_length=self.max_sequence_len - 1),
            Bidirectional(LSTM(150)),
            Dense(self.total_words, activation='softmax')
        ])
        adam = Adam(learning_rate=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    def train_model(self, xs, ys):
        self.model.fit(xs, ys, epochs=10, batch_size=128)
        joblib.dump(self.model, 'model.pkl')
        self.model.save(self.model_name)

    def run(self):
        self.preprocess_data()
        self.tokenize_text()
        xs, ys = self.prepare_sequences()
        self.build_model()
        self.train_model(xs, ys)


if __name__ == "__main__":
    dataset_path = "dataset.csv"
    model_name = "Text_auto_complete.h5"
    trainer = TrainModel(dataset_path, model_name)
    trainer.run()
"""-----------"""
