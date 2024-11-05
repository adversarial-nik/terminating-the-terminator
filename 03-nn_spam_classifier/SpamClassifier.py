import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from keras.models import load_model
import pickle

class SpamClassifier:

    def __init__(self):
        self.model = load_model('./model/trained_model_backup')
        self.vect = pickle.load(open('./model/vect_backup','rb'))
        self._ = self.classify('init the model')

        # self.new_data_path = './new_data.csv'
        self.new_data = []

    def classify(self, input_sms):
        # print('[+] Type of input_sms:'+str(type(input_sms)))
        ip_transformed = self.vect.transform([input_sms])
        if self.model.predict([ip_transformed]) > 0.5:
            return 1
        else:
            return 0

    def re_train(self):
        new_data = np.array(self.new_data)
        new_data_x = self.vect.transform(new_data[:,0])
        new_data_y = new_data[:,1]
        print('Training...')
        self.model.fit(new_data_x, new_data_y,
                      epochs=20,
                      batch_size=128)
        self.model.save('./model/trained_model')
        print('Trained___')

    def record_feedback(self, feedback_sms, feedback_label):
        '''
        feedback = ['message  ', 'label']
        '''
        print('recording feedback')
        self.new_data.append([feedback_sms, int(feedback_label)])

clf = SpamClassifier()
prediction = clf.classify('You have won free coupon of 5000$')

if prediction:
    print('\033[91m Message is Spam \033[0m')
else:
    print('\033[92m Message is not Spam \033[0m')