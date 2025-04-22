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

def tokenize3(text):
    tokens=NaiveBayes.tokenize(text)
    words = [stemmer.stem(word) for word in tokens if word not in stop_words]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams 

train_df = pd.read_csv("./data/Q1/train.csv") 
test_df = pd.read_csv("./data/Q1/test.csv")

text_col = "Tokenized Description"
class_col = "Class Index"

train_df[text_col] = train_df["Description"].apply(tokenize3)
test_df[text_col] = test_df["Description"].apply(tokenize3)

nb = NaiveBayes()
nb.fit(train_df, smoothening=1.0)
nb.predict(train_df, text_col=text_col, predicted_col="Predicted")
nb.predict(test_df, text_col=text_col, predicted_col="Predicted")

train_accuracy = (train_df[class_col] == train_df["Predicted"]).mean() * 100
test_accuracy = (test_df[class_col] == test_df["Predicted"]).mean() * 100

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")



test_df["Predicted"].to_csv("y_pred3.csv", index=False)
test_df[class_col].to_csv("y_test3.csv", index=False)