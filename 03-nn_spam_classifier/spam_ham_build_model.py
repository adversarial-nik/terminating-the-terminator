import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tensorflow.keras.optimizers import RMSprop, Adam

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import History
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

history = History()

data = pd.read_csv("model/spam.csv", encoding='latin-1')
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})
data['label_num'] = data.label.map({'ham':0, 'spam':1})

X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label_num"], test_size = 0.2, random_state = 10)

vect = CountVectorizer()
vect.fit(X_train)
# fn = vect.get_feature_names()

X_train_df = vect.transform(X_train).toarray()
X_test_df = vect.transform(X_test).toarray()

model = Sequential()
model.add(Dense(128, input_dim=X_train_df.shape[1], activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
# change learning_rate to lr
opt = RMSprop(learning_rate=0.001)    

model.compile(loss='mse',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(X_train_df, y_train,
          epochs=10,
          batch_size=128, callbacks=[history])

score = model.evaluate(X_test_df, y_test, batch_size=128)


print(score)
print('saving model')
model.save('/tmp/trained_model')
print('saving vect')
with open('/tmp/vect', 'wb') as f:
    pickle.dump(vect, f)

plt.plot(history.history['loss'], c='r')
plt.xlabel('Eopch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# change accuracy to acc
plt.plot(history.history['accuracy'], c='r')
plt.xlabel('Eopch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()
