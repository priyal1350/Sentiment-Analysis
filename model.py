# %%
import numpy as np
import  matplotlib.pyplot as plt
import pandas as pd
data_train=pd.read_csv("train.txt",delimiter=';',names=['text','label'])
data_test=pd.read_csv("test.txt",delimiter=';',names=['text','label'])
data=pd.concat([data_train,data_test])
data.reset_index(inplace=True,drop=True)

# %%
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data['label_enc'] = labelencoder.fit_transform(data['label'])
data.rename(columns={'label':'label_desc'},inplace=True)
data.rename(columns={'label_enc':'label'},inplace=True)

# %%
import nltk
import re


# %%
from nltk.corpus import stopwords

# %%
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

# %%
def text_transform(df_col):
  corpus = []
  for item in df_col:
      review = re.sub('[^a-zA-Z]', ' ', str(item))
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
      corpus.append(' '.join(str(x) for x in review))
  return corpus

# %%

corpus=text_transform(data.text)
corpus1=corpus

# %%
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = data.label

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# %%
parameters = {'max_features': ('auto','sqrt'),
             'n_estimators': [500, 1000, 1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}
# %%
from sklearn.ensemble import RandomForestClassifier


 # %%
classifier =  RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# %%
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report


# %%
predictions=classifier.predict(X_test)
acc_score = accuracy_score(y_test,predictions)
print('Accuracy_score: ',acc_score)

# %%
def check(prediction_input):
  if prediction_input==0:
      print("anger")
  elif prediction_input==1:
      print("fear")
  elif prediction_input==2:
      print("joy")
  elif prediction_input==3:
      print("love")
  elif prediction_input==4:
      print("sadness")
  else:
    print("surprise")

def sentiment_predictor(input1):
    input1=text_transform(input1)
    transformed_input=cv.transform(input1).toarray()
    prediction=classifier.predict(transformed_input)
    return prediction
input1=["you are not what you did"]
predicct=sentiment_predictor(input1)
print(predicct)
check(predicct)


# %%
import pickle
pickle.dump(classifier,open('models.pkl','wb'))

# %%

