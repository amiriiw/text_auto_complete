"""----------------------------------------------------------------------------------------------------
well come, this is maskiiw, this is a simple project about text auto complete.
    in this file we will use the models to complete texts.
----------------------------------------------------------------------------------------------------"""
# import what we need:
import pickle  # https://docs.python.org/3/library/pickle.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from tensorflow.keras.preprocessing.sequence import pad_sequences  # https://www.tensorflow.org/guide/keras
# ---------------------------------------------------------------------------------------------------------


class AutoComplete:

    model = tf.keras.models.load_model('nextwordpredictor.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    while True:
        seed_text = input("Enter your text as in farsi:  (write 'done' to exit.)")
        if seed_text == "done":
            break
        next_words = 2
        desired_length = 1242
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], padding='pre', maxlen=desired_length)
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
