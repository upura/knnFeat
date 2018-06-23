knnFeat
====

Feature Extraction with KNN

## Description
Python implementation of feature extraction with KNN.  

The following is R implementation:  
http://davpinto.com/fastknn/articles/knn-extraction.html#understanding-the-knn-features

## Install
```
git clone git@github.com:upura/knnFeat.git
cd knnFeat
pip install -r requirements.txt
```

## Demo
### Packages for visualization
```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
```

### Data generation
```python
x0 = np.random.rand(500) - 0.5
x1 = np.random.rand(500) - 0.5
X = np.array(list(zip(x0, x1)))
y = np.array([1 if i0 * i1 > 0 else 0 for (i0, i1)  in list(zip(x0, x1))])
```

### Visualization
<img src='img/before.png'>

### Feature extraction with KNN
```python
from knnFeat import knnExtract
newX = knnExtract(X, y, k=1, holds = 5)
```

### Visualization
<img src='img/after.png'>

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[upura](https://github.com/upura)
