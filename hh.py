# импорт библиотеки
import pandas as pd

# Загрузка данных
df = pd.read_csv('./nlp/medium_articles.csv', index_col=0)

# проверка данных из csv
df.head(10)

import pymorphy2
raw_data = df.dropna(subset=['text'])
morph = pymorphy2.MorphAnalyzer()
lemm_texts_list = []
for text in raw_data['text']:
    text_lem = [morph.parse(word)[0].normal_form for word in text.split(' ')]
    if len(text_lem) <= 2:
        lemm_texts_list.append('')
        continue
    lemm_texts_list.append(' '.join(text_lem))
raw_data['text_lemm'] = lemm_texts_list
raw_data = raw_data[raw_data['text_lemm'] != '']
raw_data.head()
from sklearn.model_selection import train_test_split
X = raw_data ['text_lemm']
y = raw_data ['tags']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 42, test_size=0.3)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

logreg = Pipeline([
                ('vect', CountVectorizer(analyzer='char', ngram_range =([2,10]))),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=3,C=15, solver='saga',
                                           multi_class='multinomial',
                                           max_iter=10000,
                                           random_state=42)),
])

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
print(classification_report(y_test, y_pred))
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")

import pickle

# Save to file in the current working directory
pkl_filename = "zz.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(logreg, file)

# Load from file
with open(pkl_filename, 'rb') as file:
   pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(y_test, y_pred)
print(score)
print("Test score: {0:.2f} %".format(100 * score))