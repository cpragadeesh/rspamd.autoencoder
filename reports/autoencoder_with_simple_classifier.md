```python
import keras
from keras import layers

# This is the size of our encoded representations
encoding_dim = 128
input_dim = 526

input_row = keras.Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu', name="compression")(input_row)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(input_dim, activation='sigmoid', name="decompression")(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_row, decoded)

# Encoder: This model maps an input to its encoded representation
encoder = keras.Model(input_row, encoded)

# Decoder: This is our encoded (32-dimensional) input
encoded_input = keras.Input(shape=(encoding_dim,))
# Retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# Create the decoder model
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```


```python
import json

dataset = json.loads(open("../data/processed/dataset.txt", 'r').read())
```


```python
import numpy as np

dataset = np.array(dataset)

X = dataset[:, :-1]
y = dataset[:, -1]
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)
```


```python
# Train the autoencoder

autoencoder.fit(X_train, X_train,
                epochs=500,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=0)
```




    <tensorflow.python.keras.callbacks.History at 0x146d329e8>




```python
# Compress the training and testing vectors for spam/ham classifier

X_train_compressed = encoder.predict(X_train)
X_test_compressed = encoder.predict(X_test)
```


```python
# Train spam/ham classifier

from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=1000, activation='relu', solver='adam', batch_size=250, verbose=False).fit(X_train_compressed, y_train)
```


```python
# Perform prediction on the testing dataset
y_pred = clf.predict(X_test_compressed)

# Analyse the performance
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print ("Number of ham correctly classified: " + str(tn))
print ("Number of spam correctly classified: " + str(tp))
print ("Number of ham incorrectly classified: " + str(fp))
print ("Number of spam incorrectly classified: " + str(fn))

precision = tp / float(tp + fp)
recall = tp / float(tp + fn)

f1 = 2 * (precision * recall) / float(precision + recall)

print ("Precision: " + str(precision))
print ("Recall: " + str(recall))
print ("F1-score: " + str(f1))
```

    Number of ham correctly classified: 400
    Number of spam correctly classified: 334
    Number of ham incorrectly classified: 13
    Number of spam incorrectly classified: 15
    Precision: 0.962536023054755
    Recall: 0.9570200573065902
    F1-score: 0.9597701149425288



```python

```
