import pickle
import numpy as np

labels = ['Good query','Bad query']
# query = '/<script>alert("The bad query")</script>'
# query = '/examples/good/query'

# load model and tfidf vectorizer
lgs = pickle.load(open('pickled_lgs', 'rb'))
vectorizer = pickle.load(open('pickled_vectorizer','rb'))

while True:
    try:
        # vectorize
        query = input('Query: ')
        query_vectorized = vectorizer.transform([query])

        # predict
        proba = lgs.predict_proba(query_vectorized)

        print('Predicted probabilities: ',proba)
        print('Predicted class: ',labels[np.argmax(proba)])
        print()
    except KeyboardInterrupt:
        break