from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
maxlen = 32 
X_train = np.array([1,2,3,4,5,6,7,8,9]).reshape(3,3,1)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
print(X_train)