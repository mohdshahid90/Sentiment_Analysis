import pandas as pd
input_data = pd.read_csv("C:/Users/DELL/Downloads/Compressed/NLP-Data-sets/Dats sets/User_Reviews/User_Reviews/User_movie_review.csv")

input_data.shape
input_data.columns
input_data.head(10)

input_data['class'].value_counts()

from sklearn.feature_extraction.text import CountVectorizer
countvec1 = CountVectorizer()

dtm_v1 = pd.DataFrame(countvec1.fit_transform(input_data['text']).toarray(), columns=countvec1.get_feature_names(), index=None)
dtm_v1['class'] = input_data['class']
dtm_v1.head()

import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

stemmer = PorterStemmer()
def tokenize(text):
    text = stemmer.stem(text)
    text = re.sub(r'\W+|\d+|_', ' ', text)
    tokens = nltk.word_tokenize(text)
    return tokens

countvec = CountVectorizer(min_df=5, tokenizer = tokenize, stop_words = stopwords.words('english'))
dtm = pd.DataFrame(countvec.fit_transform(input_data['text']).toarray(), columns = countvec.get_feature_names(), index = None)
dtm['class'] = input_data['class']
dtm.head()

##Building training and testing sets

df_train = dtm[:1900]
df_test = dtm[1900:]

### Building Naive Bayes Model ###

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
X_train = df_train.drop(['class'], axis = 1)
## Fitting model to our data
clf.fit(X_train,df_train['class'])
X_test= df_test.drop(['class'], axis = 1)
clf.score(X_test, df_test['class'])

## Prediction
pred_sentiment=clf.predict(df_test.drop('class', axis=1))
print(pred_sentiment)










 



