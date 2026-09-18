import pandas as pd
import  numpy as np
import joblib # to load our saved model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

dataset = pd.read_csv("news_dataset.csv")
print("Dataset loaded")
tfidf = TfidfVectorizer(stop_words = "english",lowercase=True,max_features=100)
X = tfidf.fit_transform(dataset["text"]).toarray()
y = dataset["label"].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = joblib.load("fake_news_model.pkl") # We just import a entire model into this project [ Trained model ]
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test) # predicts as the probabilty for each class 
# predict_proba returns something like [[0.2,0.8],[0.65,0.35],...]
# one probability per class , in the order model.classes_ lists them
print("\nModel classes order:",model.classes_) # check like which column is equal to which label

results = pd.DataFrame({
    "Predicted":y_pred,
    "Actual":y_test,
    "Confidence score":np.max(y_prob,axis=1) # Higher probability higher score
})
print("\nPrediction results with confidence:\n",results.to_string(index=False)) # nicely formatted table

#baseline accuracy/Overall Accuracy
baseline_accuracy = accuracy_score(y_test,y_pred)
print(f"\nBaseline accuracy:{baseline_accuracy}")

# Tuning the model using GridSearchCV
# alpha controls the smoothing - which the model handles rare unseen words better
param_grid = {"alpha":[0.1,0.5,1.0,1.5,2.0]}
grid_search = GridSearchCV(MultinomialNB(),param_grid,cv=3) # try each value , 3 fold cross validation
grid_search.fit(X_train,y_train) # run the search
print("\nBest Alpha found:",grid_search.best_params_) # which values worked best 
print("Best cross validations accuracy:",grid_search.best_score_) # how good it is
best_model = grid_search.best_estimator_ # evaluate the best-performing model
y_pred_tuned = best_model.predict(X_test) # predict again
tuned_accuracy = accuracy_score(y_test,y_pred_tuned) # new accuracy
print(f"\nTuned model accuracy: {tuned_accuracy}") # compare to baseline
print("\nDetailed report(tuned model):\n",classification_report(y_test,y_pred_tuned,zero_division=0)) # compare to headline
joblib.dump(best_model,"fake_news_model.pkl")
joblib.dump(tfidf,"tfidf_vectorizer.pkl")
print("\nSaved improved model -> fake_news_model.pkl")
