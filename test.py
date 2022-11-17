import pickle


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression



logreg = Pipeline([
                ('vect', CountVectorizer(analyzer='char', ngram_range =([2,10]))),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=3,C=1e5, solver='saga',
                                           multi_class='multinomial',
                                           max_iter=10000,
                                           random_state=42)),
])


pkl_filename = "zz.pkl"
with open(pkl_filename, 'rb') as file:
   pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
test = ['one of the most plastic systems in your brain. Neuroplasticity describes how the brain flexibly adapts to changes in the environment or when exposed to neural damage. Stimulating the brain strengthens existing neural structures and further adds fuel to the brain’s capacity to remain adaptive, thereby keeping it young. And your smell system is particularly adept at repair and renewal. (Olfactory cells have recently been used in human transplant therapy to treat spinal cord injury,']
# получение предсказания для метода k-средних и нового набора данных
y = logreg.predict(test)
print('Предсказание: ', y)