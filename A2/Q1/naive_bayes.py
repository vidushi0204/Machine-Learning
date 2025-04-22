import numpy as np
import pandas as pd
class NaiveBayes:
    def __init__(self):
        self.class_priors = {} 
        self.word_likelihoods = {} 
        self.vocab = set() 
        self.class_counts = {} 
        self.word_counts = {} 
        self.total_words_per_class = {} 
        self.vocab_size = 0 
        self.smoothening = 1  

   
    def fit(self, df, smoothening, class_col="Class Index", text_col="Tokenized Description"):
        """
        Train the Naïve Bayes model by computing class priors and word likelihoods.

        Args:
            df (pd.DataFrame): Training data containing columns `class_col` and `text_col`
                               where `text_col` contains tokenized text.
            smoothening (float): Laplace smoothing parameter.
        """

        self.smoothening = smoothening
        total_docs = len(df)
        unique_classes = df[class_col].unique()
        
        for class_label in unique_classes:
            self.class_counts[class_label] = (df[class_col] == class_label).sum()
            self.class_priors[class_label] = np.log(self.class_counts[class_label] / total_docs)
            self.word_counts[class_label] = {}

        for _, row in df.iterrows():
            class_label = row[class_col]
            words = row[text_col]
            for word in words:
                if word not in self.word_counts[class_label]:
                    self.word_counts[class_label][word] = 0
                self.word_counts[class_label][word] += 1
                self.vocab.add(word)

        self.vocab_size = len(self.vocab)

        for class_label, word_dict in self.word_counts.items():
            self.total_words_per_class[class_label] = sum(word_dict.values())

        for class_label, word_dict in self.word_counts.items():
            total_words_in_class = self.total_words_per_class[class_label]
            self.word_likelihoods[class_label] = {}
            for word in self.vocab:
                word_freq = word_dict.get(word, 0)
                self.word_likelihoods[class_label][word] = np.log((word_freq + self.smoothening) / (total_words_in_class + self.smoothening * self.vocab_size))

    def predict(self, df, text_col="Tokenized Description", predicted_col="Predicted"):
        """
        Predict class labels for input data.

        Args:
            df (pd.DataFrame): Test data containing column `text_col` with tokenized text.
            predicted_col (str): Name of column to store predictions.
        """
        predictions = []
        
        for _, row in df.iterrows():
            words = row[text_col]
            class_scores = {}

            for class_label in self.class_priors:
                log_prob = self.class_priors[class_label]
                total_words_in_class = self.total_words_per_class[class_label]

                for word in words:
                    if word in self.vocab:
                        log_prob += self.word_likelihoods[class_label].get(
                            word, np.log(self.smoothening / (total_words_in_class + self.smoothening * self.vocab_size))
                        )

                class_scores[class_label] = log_prob
            
            predictions.append(max(class_scores, key=class_scores.get))

        df[predicted_col] = predictions
    @staticmethod 
    def tokenize(text):
        """Tokenize input text by converting to lowercase and splitting on spaces."""
        text = text.replace('.',' ').replace(',',' ').replace(';',' ')
        return text.lower().split(' ')
        
