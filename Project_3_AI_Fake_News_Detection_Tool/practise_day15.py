import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

X = np.load("X_tfidf.npy")
y = np.load("y_labels.npy",allow_pickle=True) # Load the matrix from the previous day
print("Loaded feature matrix shape:",X.shape)
print("Labels:",y)

# Splitting into  the training and test set
X_train,X_test,y_train,y_test = train_test_split(
X,y,test_size=0.2,random_state=2
)
# TF_IDF produced word-Frequency table , this data is fed to the Naive Bayes algorithm
'''
P(class | text) = [ P(text | class) × P(class) ] / P(text)
P(class | text) — probability the text belongs to a class (what we want)
P(text | class) — likelihood of seeing these words given the class
P(class) — how common that class is overall (prior)
P(text) — probability of the text itself (same for all classes, so ignored in comparison)
it multiplies these probabilities together across all words, then picks whichever class (real/fake) has the higher combined probability.
'''
model = MultinomialNB() 
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("\nPredicted Labels:",y_pred)
print("\nActual Labels:",y_test)
accuracy = accuracy_score(y_test,y_pred) # compare the predictions and gives the accuracy as score
print("\nAccuracy on the test set:",accuracy)
print("\nDetailed Report:\n",classification_report(y_test,y_pred,zero_division=0))

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
dataset = pd.read_csv("news_raw.csv")
tfidf = TfidfVectorizer(stop_words="english",lowercase=True,max_features=100)
tfidf.fit(dataset["text"]) # refit and update the articles region
new_headline = ["Shocking secret doctors dont want u to know"]
new_vector = tfidf.transform(new_headline).toarray() # Stores the vectors as np array
prediction = model.predict(new_vector) # model knows only numbers hence the tfidf transforms the words into numbers then make the predictions
print(f"\nNew Headline:{new_headline[0]}\nPredicted Label:{prediction[0]}")
