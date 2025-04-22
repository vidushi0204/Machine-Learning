import pandas as pd
from naive_bayes import NaiveBayes 
from wordcloud import WordCloud
import matplotlib.pyplot as plt


train_df = pd.read_csv("./data/Q1/train.csv") 
test_df = pd.read_csv("./data/Q1/test.csv") 

text_col = "Tokenized Description"
class_col = "Class Index"


nb = NaiveBayes()
train_df[text_col] = train_df["Description"].apply(NaiveBayes.tokenize)
test_df[text_col] = test_df["Description"].apply(NaiveBayes.tokenize)

nb.fit(train_df, smoothening=1.0)
nb.predict(train_df)
nb.predict(test_df)


train_accuracy = (train_df[class_col] == train_df["Predicted"]).mean() * 100
test_accuracy = (test_df[class_col] == test_df["Predicted"]).mean() * 100

print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

for class_label in range(1, 5): 
    word_freq = nb.word_counts.get(class_label, {})
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for Class {class_label}")
    plt.show()

test_df["Predicted"].to_csv("y_pred1.csv", index=False)
test_df[class_col].to_csv("y_test1.csv", index=False)