import pandas as pd
from naive_bayes import NaiveBayes
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def best_tokenize(text):
    tokens = NaiveBayes.tokenize(text)
    words = [stemmer.stem(word) for word in tokens if word not in stop_words]
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    return words + bigrams

train_df = pd.read_csv("./data/Q1/train.csv")
test_df = pd.read_csv("./data/Q1/test.csv")

class_col = "Class Index"

train_df["Title Tokenized"] = train_df["Title"].apply(best_tokenize)
train_df["Description Tokenized"] = train_df["Description"].apply(best_tokenize)

test_df["Title Tokenized"] = test_df["Title"].apply(best_tokenize)
test_df["Description Tokenized"] = test_df["Description"].apply(best_tokenize)

class NaiveBayesSeparate:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_probs = {} 
        self.title_probs = {} 
        self.desc_probs = {}  
        self.vocab_title = set()
        self.vocab_desc = set()

    def fit(self, df, class_col, title_col, desc_col):
        class_counts = df[class_col].value_counts().to_dict()
        total_docs = len(df)

        self.class_probs = {
            c: (class_counts[c] + self.smoothing) / 
               (total_docs + len(class_counts) * self.smoothing) 
            for c in class_counts
        }

        title_word_counts = {c: {} for c in class_counts}
        desc_word_counts = {c: {} for c in class_counts}
        title_total_counts = {c: 0 for c in class_counts}
        desc_total_counts = {c: 0 for c in class_counts}

        for _, row in df.iterrows():
            c = row[class_col]
            title_tokens = row[title_col]
            desc_tokens = row[desc_col]

            for word in title_tokens:
                self.vocab_title.add(word)
                title_word_counts[c][word] = title_word_counts[c].get(word, 0) + 1
                title_total_counts[c] += 1

            for word in desc_tokens:
                self.vocab_desc.add(word)
                desc_word_counts[c][word] = desc_word_counts[c].get(word, 0) + 1
                desc_total_counts[c] += 1

        self.title_probs = {
            c: {word: (title_word_counts[c].get(word, 0) + self.smoothing) /
                    (title_total_counts[c] + self.smoothing * len(self.vocab_title))
                for word in self.vocab_title}
            for c in class_counts
        }

        self.desc_probs = {
            c: {word: (desc_word_counts[c].get(word, 0) + self.smoothing) /
                    (desc_total_counts[c] + self.smoothing * len(self.vocab_desc))
                for word in self.vocab_desc}
            for c in class_counts
        }

    def predict(self, df, title_col, desc_col, predicted_col):
        predictions = []
        for _, row in df.iterrows():
            title_tokens = row[title_col]
            desc_tokens = row[desc_col]

            class_scores = {}
            for c in self.class_probs:
                class_scores[c] = np.log(self.class_probs[c])  

                for word in title_tokens:
                    if word in self.title_probs[c]:
                        class_scores[c] += np.log(self.title_probs[c][word])

                for word in desc_tokens:
                    if word in self.desc_probs[c]:
                        class_scores[c] += np.log(self.desc_probs[c][word])

            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        df[predicted_col] = predictions

nb_separate = NaiveBayesSeparate(smoothing=1.0)
nb_separate.fit(train_df, class_col=class_col, title_col="Title Tokenized", desc_col="Description Tokenized")

nb_separate.predict(train_df, title_col="Title Tokenized", desc_col="Description Tokenized", predicted_col="Predicted")
nb_separate.predict(test_df, title_col="Title Tokenized", desc_col="Description Tokenized", predicted_col="Predicted")

train_accuracy = (train_df[class_col] == train_df["Predicted"]).mean()
test_accuracy = (test_df[class_col] == test_df["Predicted"]).mean()

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
