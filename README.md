# Relation extraction

## Prerequisites

- pip
- python 3
- pipenv

install dependencies
```
pipenv install
```

## SVM

```
from svm import Dataset

d = Dataset("./train.txt")
d.train()
d.evaluate()
```

## BERT

Open bert.ipynb
Execute from start to end