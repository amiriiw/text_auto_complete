"""----------------------------------------------------------------------------------------------------
well come, this is maskiiw, this is a simple project about text auto complete.
    in this file we will train the persian data set to use as auto complete model.
----------------------------------------------------------------------------------------------------"""
# import what we need:
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
# -----------------------------------------------------------------------------------------------------------------


class TrainModel:

    dataset = pd.read_csv('data.csv')
    dataset.head()
    print("records: ", dataset.shape[0])
    print("fields: ", dataset.shape[1])
    dataset['Text'] = dataset['Text'].apply(lambda x: x.replace(u'\xa0', u' '))
    dataset['Text'] = dataset['Text'].apply(lambda x: x.replace('\u200a', ' '))
    tokenizer = Tokenizer(oov_token='<oov>')
    tokenizer.fit_on_texts(dataset['Text'])
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    total_words = len(tokenizer.word_index) + 1

    input_sequences = []
    for line in dataset['Text']:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    print("Total input sequences: ", len(input_sequences))

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(150)))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    model.fit(xs, ys, epochs=10, batch_size=128)
    joblib.dump(model, 'model.pkl')
    model.save('nextwordpredictor.h5')
# ------------------------------------

    model = tf.keras.models.load_model('nextwordpredictor.h5')
    while True:
        seed_text = input("Enter your text as in farsi:")
        if seed_text == "0":
            break
        next_words = 10
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]
            predicted_index = np.argmax(predicted_probs)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
            seed_text += " " + output_word
        print(seed_text)
# ----------------------
