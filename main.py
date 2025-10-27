import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Layer, Bidirectional # type: ignore
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from transformers import BertTokenizer, TFBertModel
import pandas as pd
import time

# Check if GPU is available
device = "GPU" 
print(f"Training on: {device}")

# Load Dataset
train_data = pd.read_csv("./dataset/train_preprocessed.csv")
test_data = pd.read_csv("./dataset/test_preprocessed.csv")

# Clean NaN
train_data['REVIEW_TEXT'] = train_data['REVIEW_TEXT'].astype(str).fillna("")
test_data['REVIEW_TEXT'] = test_data['REVIEW_TEXT'].astype(str).fillna("")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 256  # Increased length

def encode_texts(texts, max_len=MAX_LEN):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    encoding = tokenizer(texts, max_length=max_len, padding='max_length', truncation=True, return_tensors="np")
    return encoding["input_ids"], encoding["attention_mask"]

X_train_ids, X_train_mask = encode_texts(train_data['REVIEW_TEXT'])
X_test_ids, X_test_mask = encode_texts(test_data['REVIEW_TEXT'])

# Numeric features
numeric_columns = [
    'AVERAGE_RATING', 'RATING_DEVIATION', 'TOTAL_PRODUCT_REVIEWS',
    'REVIEW_LENGTH', 'RATING_CATEGORY', 'SINGLE_RATING_CATEGORY',
    'REVIEW_COUNT_DATE', 'SAME_DATE_MULTIPLE_REVIEWS', 'MAX_USER_REVIEWS_DAY',
    'TIMESTAMP_DIFFERENCE', 'AVERAGE_USER_REVIEW_LENGTH', 'TOTAL_USER_REVIEWS',
    'PERCENTAGE_POSITIVE_REVIEWS', 'RATIO_POSITIVE_NEGATIVE'
]

X_train_numeric = train_data[numeric_columns].astype(float).values
X_test_numeric = test_data[numeric_columns].astype(float).values
y_train = train_data['LABEL'].values
y_test = test_data['LABEL'].values

# Load & fine-tune BERT
bert_model = TFBertModel.from_pretrained('bert-base-uncased')
bert_model.trainable = True  # Now fine-tuning BERT

# BERT Layer
@tf.keras.utils.register_keras_serializable()
class BertLayer(Layer):
    def __init__(self, model_name="bert-base-uncased", **kwargs):
        self.model_name = model_name
        self.bert = TFBertModel.from_pretrained(model_name)
        super(BertLayer, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

    @classmethod
    def from_config(cls, config):
        # Handle old config saved as {"bert_model": "tf_bert_model"}
        if "bert_model" in config:
            config["model_name"] = "bert-base-uncased"  # or use logic to detect proper model
            del config["bert_model"]
        return cls(**config)

# Model Architecture
def create_bert_lstm_model(bert_model, max_len=MAX_LEN, num_numeric_features=14):
    input_ids = Input(shape=(max_len,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(max_len,), dtype=tf.int32, name='attention_mask')
    numeric_input = Input(shape=(num_numeric_features,), name='numeric_input')

    bert_output = BertLayer(bert_model)([input_ids, attention_mask])
    lstm_output = Bidirectional(LSTM(64, return_sequences=False))(bert_output)

    numeric_dense = Dense(64, activation='relu')(numeric_input)

    merged = Concatenate()([lstm_output, numeric_dense])
    x = Dropout(0.4)(merged)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_ids, attention_mask, numeric_input], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Build model
model = create_bert_lstm_model("bert-base-uncased")

# Callbacks
checkpoint_path = "./models/fake_review_model.keras"
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True),
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True)
]

# Train
start_time = time.time()
history = model.fit(
    [X_train_ids, X_train_mask, X_train_numeric], y_train,
    validation_data=([X_test_ids, X_test_mask, X_test_numeric], y_test),
    epochs=3,
    batch_size=32,
    callbacks=callbacks
)
end_time = time.time()
print(f"Training Time: {(end_time - start_time)/60:.2f} minutes")

# Final Accuracy
print(f"Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")

# Save Model
model.save("./models/fake_reviews_model_2.keras")
print("Model saved successfully!")