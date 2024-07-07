"""----------------------------------------------------------------------------------------------------
well come, this is amiriiw, this is a simple project about text auto complete.
    in this file we will use the models to complete texts.
----------------------------------------------------------------------------------------------------"""
# import what we need:
import pickle  # https://docs.python.org/3/library/pickle.html
import numpy as np  # https://numpy.org/devdocs/user/absolute_beginners.html
import tensorflow as tf  # https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html
from tensorflow.keras.preprocessing.sequence import pad_sequences  # https://www.tensorflow.org/guide/keras
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog
# --------------------------------------------------------------------------------------------------------------------


class AutoCompleteApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initui()

        self.seed_text_label = None
        self.seed_text_input = None
        self.complete_button = None
        self.output_text_label = None
        self.output_text_area = None

        self.model = tf.keras.models.load_model(self.get_model_path())
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

    def initui(self):
        self.setWindowTitle('Text Auto-Completion')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.seed_text_label = QLabel('Enter your text in Farsi:')
        layout.addWidget(self.seed_text_label)

        self.seed_text_input = QLineEdit()
        layout.addWidget(self.seed_text_input)

        self.complete_button = QPushButton('Complete Text')
        self.complete_button.clicked.connect(self.complete_text)
        layout.addWidget(self.complete_button)

        self.output_text_label = QLabel('Completed Text:')
        layout.addWidget(self.output_text_label)

        self.output_text_area = QTextEdit()
        self.output_text_area.setReadOnly(True)
        layout.addWidget(self.output_text_area)

        self.setLayout(layout)

    def get_model_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "",
                                                    "Model Files (*.h5 *.pb *.onnx);;All Files (*)", options=options)
        return model_path

    def complete_text(self):
        seed_text = self.seed_text_input.text()

        if seed_text:
            next_words = 2
            desired_length = 1242
            completed_text = seed_text

            for _ in range(next_words):
                token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
                token_list = pad_sequences([token_list], padding='pre', maxlen=desired_length)
                predicted_probs = self.model.predict(token_list, verbose=0)[0]
                predicted_index = np.argmax(predicted_probs)

                output_word = ""
                for word, index in self.tokenizer.word_index.items():
                    if index == predicted_index:
                        output_word = word
                        break

                seed_text += " " + output_word
                completed_text += " " + output_word

            self.output_text_area.setPlainText(completed_text)

        else:
            self.output_text_area.setPlainText("Please enter some text to complete.")


if __name__ == '__main__':
    app = QApplication([])
    autocomplete_app = AutoCompleteApp()
    autocomplete_app.show()
    app.exec_()
# --------------
