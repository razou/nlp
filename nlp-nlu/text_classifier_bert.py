import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer, TFBertForSequenceClassification


class BertTextClassifier:

    def load_data(self):
        # Load IMDB movie review dataset
        train_data, validation_data = tfds.load(
            name="imdb_reviews",
            split=('train[:60%]', 'train[60%:]'),
            as_supervised=True
        )
        return train_data, validation_data

    @staticmethod
    def text_encoder(text, label):
        """
        Tokenize and encode dataset

        :param text: Input datasets
        :param label: Class label
        :return: preprocessed datasets
        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        input_ids = tokenizer.encode(text.numpy().decode("utf-8"), add_special_tokens=True, max_length=512)
        return input_ids, label

    def encode_examples(self, text, label):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def encode(text, label):
            text_string = text.numpy().decode('utf-8')  # Convert byte string to string
            input_ids = tokenizer.encode(text_string, add_special_tokens=True, max_length=512,
                                         padding='max_length', truncation=True)  # Pad or truncate sequences
            return input_ids, label

        input_ids, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int32, tf.int64))
        input_ids.set_shape([512])  # Set fixed shape after padding/truncation
        label.set_shape([])

        return input_ids, label

    def data_preprocessing(self, data, shuffle: bool = False):
        # encoded_data = datasets.map(self.text_encoder, num_parallel_calls=tf.datasets.experimental.AUTOTUNE)
        encoded_data = data.map(self.encode_examples, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        #if shuffle:
        #    return encoded_data.shuffle(1000).batch(32).prefetch(tf.datasets.experimental.AUTOTUNE)
        #else:
        return encoded_data.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    def create_model(self, learning_rate: float, metric_name: str):
        """
        Load Bert pretrained model, specify parameters, metric and loss function.

        :param learning_rate: Learning rate hyperparameter
        :param metric_name: Evaluation metric
        :return: Compiled Model
        """
        model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy(metric_name)
        model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        return model

    def train(self, **kwargs):
        leaning_rate = float(kwargs.get("leaning_rate", 3e-5))
        metric_name = kwargs.get("metric_name", "accuracy")

        model = self.create_model(learning_rate=leaning_rate, metric_name=metric_name)
        raw_train_data, raw_validation_data = self.load_data()
        print('****************** load datasets: Ok ***********************')

        preprocessed_train_data = self.data_preprocessing(data=raw_train_data, shuffle=True)
        preprocessed_validation_data = self.data_preprocessing(data=raw_validation_data)
        print('****************** data_preprocessing: Ok *********************')

        model.fit(preprocessed_train_data, epochs=10, validation_data=preprocessed_validation_data)

        # Evaluate the model
        results = model.evaluate(preprocessed_validation_data)
        print("Validation Accuracy:", results[1])

    def __call__(self, *args, **kwargs):
        self.train()


if __name__ == "__main__":
    BertTextClassifier()()
