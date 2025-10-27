from predictor import predict_fake_review

test_review = "This product changed my life! Highly recommend it."
label, confidence = predict_fake_review(test_review)
print(f"Prediction: {label} (Confidence: {confidence:.2f}%)")