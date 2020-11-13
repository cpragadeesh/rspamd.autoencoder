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
print "Number of emails: " + str(y.shape[0])
print "Number of spam: " + str(y[y==1].shape[0])
print "Number of ham: " + str(y[y==-1].shape[0])

print "Number of unique symbols: " + str(X.shape[1])
```

    Number of emails: 3048
    Number of spam: 1397
    Number of ham: 1651
    Number of unique symbols: 526



```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                    random_state=1)

print "Training samples: " + str(X_train.shape[0])
print "Testing samples: " + str(X_test.shape[0])
```

    Training samples: 2286
    Testing samples: 762



```python
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(500,), activation='relu', solver='adam', batch_size=100, verbose=True).fit(X_train, y_train)
```

    Iteration 1, loss = 0.37712267
    Iteration 2, loss = 0.17747624
    Iteration 3, loss = 0.13422646
    Iteration 4, loss = 0.11804128
    Iteration 5, loss = 0.10495433
    Iteration 6, loss = 0.09363893
    Iteration 7, loss = 0.08895384
    Iteration 8, loss = 0.08058484
    Iteration 9, loss = 0.07314565
    Iteration 10, loss = 0.06887500
    Iteration 11, loss = 0.06403496
    Iteration 12, loss = 0.05886320
    Iteration 13, loss = 0.05562595
    Iteration 14, loss = 0.05404621
    Iteration 15, loss = 0.04851559
    Iteration 16, loss = 0.04559954
    Iteration 17, loss = 0.04369478
    Iteration 18, loss = 0.04140317
    Iteration 19, loss = 0.03726862
    Iteration 20, loss = 0.03597393
    Iteration 21, loss = 0.03349803
    Iteration 22, loss = 0.03108009
    Iteration 23, loss = 0.02997789
    Iteration 24, loss = 0.02799606
    Iteration 25, loss = 0.02727137
    Iteration 26, loss = 0.02548459
    Iteration 27, loss = 0.02548111
    Iteration 28, loss = 0.02404882
    Iteration 29, loss = 0.02283011
    Iteration 30, loss = 0.02230326
    Iteration 31, loss = 0.02067667
    Iteration 32, loss = 0.02125572
    Iteration 33, loss = 0.02000329
    Iteration 34, loss = 0.01805838
    Iteration 35, loss = 0.01840314
    Iteration 36, loss = 0.01748564
    Iteration 37, loss = 0.01668973
    Iteration 38, loss = 0.01665202
    Iteration 39, loss = 0.01616708
    Iteration 40, loss = 0.01559568
    Iteration 41, loss = 0.01546501
    Iteration 42, loss = 0.01456760
    Iteration 43, loss = 0.01412628
    Iteration 44, loss = 0.01516742
    Iteration 45, loss = 0.01364081
    Iteration 46, loss = 0.01354670
    Iteration 47, loss = 0.01274093
    Iteration 48, loss = 0.01345360
    Iteration 49, loss = 0.01374831
    Iteration 50, loss = 0.01345989
    Iteration 51, loss = 0.01170683
    Iteration 52, loss = 0.01138362
    Iteration 53, loss = 0.01193861
    Iteration 54, loss = 0.01096610
    Iteration 55, loss = 0.01097960
    Iteration 56, loss = 0.01195822
    Iteration 57, loss = 0.01485385
    Iteration 58, loss = 0.01258459
    Iteration 59, loss = 0.01214198
    Iteration 60, loss = 0.01133402
    Iteration 61, loss = 0.01055753
    Iteration 62, loss = 0.00949574
    Iteration 63, loss = 0.01061503
    Iteration 64, loss = 0.01025458
    Iteration 65, loss = 0.00936592
    Iteration 66, loss = 0.01201895
    Iteration 67, loss = 0.01090021
    Iteration 68, loss = 0.01010509
    Iteration 69, loss = 0.00927248
    Iteration 70, loss = 0.00993236
    Iteration 71, loss = 0.00937250
    Iteration 72, loss = 0.00972058
    Iteration 73, loss = 0.00888179
    Iteration 74, loss = 0.00926341
    Iteration 75, loss = 0.00944697
    Iteration 76, loss = 0.01023403
    Iteration 77, loss = 0.01017100
    Iteration 78, loss = 0.00848673
    Iteration 79, loss = 0.00807941
    Iteration 80, loss = 0.00870413
    Iteration 81, loss = 0.00807366
    Iteration 82, loss = 0.00884568
    Iteration 83, loss = 0.00983048
    Iteration 84, loss = 0.00875041
    Iteration 85, loss = 0.00793485
    Iteration 86, loss = 0.00812260
    Iteration 87, loss = 0.00809863
    Iteration 88, loss = 0.00825790
    Iteration 89, loss = 0.00836431
    Iteration 90, loss = 0.00827144
    Iteration 91, loss = 0.00862473
    Iteration 92, loss = 0.00764116
    Iteration 93, loss = 0.00812574
    Iteration 94, loss = 0.00803432
    Iteration 95, loss = 0.00865225
    Iteration 96, loss = 0.00787842
    Iteration 97, loss = 0.00870834
    Iteration 98, loss = 0.00765004
    Iteration 99, loss = 0.01026460
    Iteration 100, loss = 0.00957423
    Iteration 101, loss = 0.00817809
    Iteration 102, loss = 0.00780187
    Iteration 103, loss = 0.00794922
    Training loss did not improve more than tol=0.000100 for 10 consecutive epochs. Stopping.



```python
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print "Number of ham correctly classified: " + str(tn)
print "Number of spam correctly classified: " + str(tp)
print "Number of ham incorrectly classified: " + str(fp)
print "Number of spam incorrectly classified: " + str(fn)

precision = tp / float(tp + fp)
recall = tp / float(tp + fn)

f1 = 2 * (precision * recall) / float(precision + recall)

print ("Precision: " + str(precision))
print ("Recall: " + str(recall))
print ("F1-score: " + str(f1))
```

    Number of ham correctly classified: 402
    Number of spam correctly classified: 333
    Number of ham incorrectly classified: 11
    Number of spam incorrectly classified: 16
    Precision: 0.9680232558139535
    Recall: 0.9541547277936963
    F1-score: 0.9610389610389611



```python

```
