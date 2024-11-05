import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from PIL import Image
from keras import optimizers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras import Sequential
from keras.layers import Dense, Dropout, Conv2D, AveragePooling2D, MaxPooling2D, Flatten
from keras.callbacks import History
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
tf.config.set_visible_devices([], 'GPU')

history = History()

dv = pd.read_csv('devanagari_prime_digits.csv', header=None).values

X_train, X_test, y_train, y_test = train_test_split(dv[:,:-1],dv[:,-1], test_size = 0.2, random_state = 10)

lb = LabelBinarizer()
lb.fit(dv[:,-1])

y_test = lb.transform(y_test)
y_train = lb.transform(y_train)


x_train_conv = np.array([_.reshape(32,32,1) for _ in X_train])
x_test_conv = np.array([_.reshape(32,32,1) for _ in X_test])

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), activation='sigmoid', input_shape=(32,32,1)))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# model.add(Dense(1000, activation='relu'))
model.add(Dense(4, activation='softmax'))
opt = optimizers.RMSprop(learning_rate=0.0005)
model.compile(loss='mse',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(x_train_conv, y_train,validation_data=(x_test_conv,y_test),
          epochs=10,
          batch_size=128, callbacks=[history])

plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('mse')
plt.subplot(1,2,2)
# change accuracy to acc
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

model.save('/tmp/devnagri_model.h5')


