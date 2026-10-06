###############################################
# Step 1: Import required libraries
###############################################

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


###############################################
# Step 2: Configuration of values
###############################################

VOCAB_SIZE = 10000
MAX_LENGTH = 200


###############################################
# Step 3: Load the IMDB dataset
###############################################

print("-" * 40)
print("Movie Review Sentiment Analysis using LSTM")
print("-" * 40)

print("Loading the dataset...")

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(
    num_words=VOCAB_SIZE
)

print("IMDB dataset loaded successfully")
print("Number of training reviews:", len(X_train))
print("Number of testing reviews:", len(X_test))


###############################################
# X_train -> Training reviews
# Y_train -> Training sentiments
# X_test  -> Testing reviews
# Y_test  -> Testing sentiments
#
# 0 -> Negative
# 1 -> Positive
###############################################


###############################################
# Step 4: Load the word dictionary
###############################################

word_index = imdb.get_word_index()


###############################################
# Step 5: Create reverse dictionary
###############################################

reverse_word_index = {}

for word, index in word_index.items():
    reverse_word_index[index + 3] = word


###############################################
# Step 6: Function to decode the review
###############################################

def decode_review(encoded_review):

    words = []

    for number in encoded_review:

        if number >= 3:
            word = reverse_word_index.get(number, "?")
            words.append(word)

    return " ".join(words)


###############################################
# Step 7: Display sample reviews
###############################################

print("-" * 40)
print("--------- Sample Reviews ---------")
print("-" * 40)

for i in range(3, 7):

    review = decode_review(X_train[i])

    print("-" * 40)
    print("Review number:", i + 1)
    print("Review:")
    print(review)
    print("-" * 40)

    if Y_train[i] == 1:
        print("Sentiment: POSITIVE")
    else:
        print("Sentiment: NEGATIVE")


###############################################
# Step 8: Padding
###############################################

X_train_padded = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH
)

X_test_padded = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH
)

print("Training data shape:", X_train_padded.shape)
print("Testing data shape:", X_test_padded.shape)


###############################################
# Step 9: Create LSTM model
###############################################

model = Sequential()

model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=32
    )
)

model.add(
    LSTM(
        units=64
    )
)

model.add(
    Dense(
        units=1,
        activation="sigmoid"
    )
)


###############################################
# Project Architecture:
#
# Review
#   ↓
# Embedding
#   ↓
# LSTM
#   ↓
# Dense
#   ↓
# Sigmoid
#   ↓
# Positive / Negative
###############################################


###############################################
# Step 10: Compile the model
###############################################

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully")


###############################################
# Step 11: Train the model
###############################################

print("Model training started...")

history = model.fit(
    X_train_padded,
    Y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

print("Model training completed")


###############################################
# Step 12: Plot training and validation accuracy
###############################################

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "accuracy_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


###############################################
# Step 13: Plot training and validation loss
###############################################

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "loss_graph.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


###############################################
# Step 14: Evaluate the model
###############################################

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    Y_test,
    verbose=0
)

print("Testing loss:", test_loss)
print("Testing accuracy:", test_accuracy)


###############################################
# Step 15: Predict a test review
###############################################

TEST_REVIEW_NUMBER = 0

original_review = X_test[TEST_REVIEW_NUMBER]

decoded_review = decode_review(original_review)

print("-" * 40)
print("Review given to the model:")
print(decoded_review)


###############################################
# Step 16: Get the actual sentiment
###############################################

actual_value = Y_test[TEST_REVIEW_NUMBER]

if actual_value == 1:
    actual_sentiment = "POSITIVE"
else:
    actual_sentiment = "NEGATIVE"

print("Actual sentiment:", actual_sentiment)


###############################################
# Step 17: Predict the sentiment
###############################################

review_for_prediction = X_test_padded[
    TEST_REVIEW_NUMBER:TEST_REVIEW_NUMBER + 1
]

prediction = model.predict(
    review_for_prediction,
    verbose=0
)

probability = prediction[0][0]

if probability >= 0.5:
    predicted_sentiment = "POSITIVE"
else:
    predicted_sentiment = "NEGATIVE"

print("-" * 40)
print("Final Result")
print("-" * 40)

print("Prediction probability:", probability)
print("Actual sentiment:", actual_sentiment)
print("Predicted sentiment:", predicted_sentiment)

print("-" * 40)


###############################################
# Step 18: Predict custom review
###############################################

print("\n" + "-" * 40)
print("Custom Movie Review Prediction")
print("-" * 40)

custom_review = input("Enter your movie review: ")

words = custom_review.lower().split()

encoded_review = []

for word in words:

    index = word_index.get(word)

    if index is not None and index < VOCAB_SIZE:
        encoded_review.append(index + 3)
    else:
        encoded_review.append(2)


###############################################
# Pad the custom review
###############################################

custom_review_padded = pad_sequences(
    [encoded_review],
    maxlen=MAX_LENGTH
)


###############################################
# Predict custom review sentiment
###############################################

prediction = model.predict(
    custom_review_padded,
    verbose=0
)

probability = prediction[0][0]


###############################################
# Display result
###############################################

if probability >= 0.5:
    sentiment = "POSITIVE"
else:
    sentiment = "NEGATIVE"

print("\nReview:", custom_review)
print("Sentiment:", sentiment)
print("Probability:", probability)
print("-" * 40)