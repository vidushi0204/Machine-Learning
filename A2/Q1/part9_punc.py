import pandas as pd
from naive_bayes import NaiveBayes
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def best_tokenize_title(text):
    tokens = NaiveBayes.tokenize(text)
    words = [stemmer.stem(word) for word in tokens if word not in stop_words]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

def best_tokenize_description(text):
    tokens = NaiveBayes.tokenize(text)
    words = [stemmer.stem(word) for word in tokens if word not in stop_words]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

def count_punctuation(text):
    punctuation_marks = '.,!?;:\'"()-'
    return sum(1 for char in text if char in punctuation_marks)

def add_punctuation_feature(text):
    count = count_punctuation(text)
    return ["PUNCTUATION"] * count

train_df = pd.read_csv("./data/Q1/train.csv")
test_df = pd.read_csv("./data/Q1/test.csv")

class_col = "Class Index"

train_df["Merged Text"] = (
    train_df["Title"].apply(best_tokenize_title) +
    train_df["Description"].apply(best_tokenize_description) +
    train_df["Title"].apply(add_punctuation_feature) +
    train_df["Description"].apply(add_punctuation_feature)
)

test_df["Merged Text"] = (
    test_df["Title"].apply(best_tokenize_title) +
    test_df["Description"].apply(best_tokenize_description) +
    test_df["Title"].apply(add_punctuation_feature) +
    test_df["Description"].apply(add_punctuation_feature)
)

nb = NaiveBayes()
nb.fit(train_df, smoothening=1.0, class_col=class_col, text_col="Merged Text")
nb.predict(train_df, text_col="Merged Text", predicted_col="Predicted")
nb.predict(test_df, text_col="Merged Text", predicted_col="Predicted")

train_accuracy = (train_df[class_col] == train_df["Predicted"]).mean()
test_accuracy = (test_df[class_col] == test_df["Predicted"]).mean()

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

test_df["Predicted"].to_csv("y_pred6.csv", index=False)
test_df[class_col].to_csv("y_test6.csv", index=False)
