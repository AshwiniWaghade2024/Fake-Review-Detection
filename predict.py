import numpy as np
from transformers import BertTokenizer
from tensorflow.keras.models import load_model
from utils.bert_layer import BertLayer

# Load model and tokenizer only once
model = load_model("./model/fake_reviews_model.keras", custom_objects={'BertLayer': BertLayer})
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def predict_fake_review(review_text, max_len=256):
    encodings = tokenizer(
        review_text,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    numeric_features = np.zeros((1, 14))

    prediction = model.predict([input_ids, attention_mask, numeric_features])
    label = "Fake Review" if prediction[0][0] > 0.7 else "Genuine Review"
    confidence = prediction[0][0] * 100 if prediction[0][0] > 0.5 else (1 - prediction[0][0]) * 100
    return label, confidence