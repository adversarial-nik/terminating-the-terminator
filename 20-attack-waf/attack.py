import pickle
import numpy as np
import tqdm

labels = ['Good query','Bad query']

# load model and tfidf vectorizer
lgs = pickle.load(open('pickled_lgs', 'rb'))
vectorizer = pickle.load(open('pickled_vectorizer','rb'))

def score(query):
    query_vectorized = vectorizer.transform([query])
    proba = lgs.predict_proba(query_vectorized)
    good_qscore = proba[0][0]
    print('Predicted class: ',labels[np.argmax(proba)])
    print('Predicted probabilities: ',proba)
    score = (good_qscore * 100) / len(query)
    return score

query = '<script>alert("123")</script>'
print("CTF Score", score(query))
