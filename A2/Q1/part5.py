import pandas as pd
from naive_bayes import NaiveBayes
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def tokenize1(text):
    text = text.replace('.', ' ').replace(',', ' ').replace(';', ' ')
    return text.lower().split(' ')

def tokenize2(text):
    tokens = NaiveBayes.tokenize(text)
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def tokenize3(text):
    tokens = NaiveBayes.tokenize(text)
    words = [stemmer.stem(word) for word in tokens if word not in stop_words]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

train_df = pd.read_csv("./data/Q1/train.csv")
test_df = pd.read_csv("./data/Q1/test.csv")

class_col = "Class Index"
results = {}

for i, tokenize_fn in enumerate([tokenize1, tokenize2, tokenize3], start=1):
    text_col = f"Tokenized Title {i}"
    
    train_df[text_col] = train_df["Title"].apply(tokenize_fn)
    test_df[text_col] = test_df["Title"].apply(tokenize_fn)

    nb = NaiveBayes()
    nb.fit(train_df, smoothening=1.0, class_col=class_col, text_col=text_col)
    nb.predict(train_df, text_col=text_col, predicted_col="Predicted")
    nb.predict(test_df, text_col=text_col, predicted_col="Predicted")

    train_accuracy = (train_df[class_col] == train_df["Predicted"]).mean()
    test_accuracy = (test_df[class_col] == test_df["Predicted"]).mean()

    results[f"Tokenize{i}"] = (train_accuracy, test_accuracy)

    test_df[[class_col]].to_csv(f"y_test{i}.csv", index=False, header=False)
    test_df[["Predicted"]].to_csv(f"y_pred{i}.csv", index=False, header=False)

print("Accuracy")
for key, (train_acc, test_acc) in results.items():
    print(f"{key}: Train = {train_acc:.4f}, Test = {test_acc:.4f}")
