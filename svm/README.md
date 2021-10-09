# Eye Blink using SVM

The SVM uses eye aspect ratio (ear) in consecutive video frames to predict eye blink.

## Data

Eyeblink8 is used to train.

* [Mendele data](https://data.mendeley.com/datasets/9yzdnng594/1).
* [eyeblink8](https://www.blinkingmatters.com/research)

## Preprocess

Get training data ready

## Train

Be sure training data is prepared in `eyeblink8` and run:

```bash
python3 preprocess.py

```

Then train svm:

```python
from utils import *
from utils_frame_based import *
from train import *

# uses 7 frames to predict
X, y = build_svm_data('./train/training_set.pkl', 7)

norm_X = transform_svm_data(X)

# adjust on your need
params = {
    'kernel': ['rbf'],
    'C': [10],
    'gamma': ['scale'],
    'max_iter': [5000],
    'class_weight': [None],
}
svm = train(norm_X, y, params)

save_model(svm, 'svm_7.pkl')
```

## Reference

Largely modified from Mustafa A. Hakkoz's kaggle:

* [Part 1](https://www.kaggle.com/hakkoz/eye-blink-detection-3-ml-model-part1)
* [Part 2](https://www.kaggle.com/hakkoz/eye-blink-detection-3-ml-model-part2)
