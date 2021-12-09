# KoSentimentAnalysis

### 왜 한국어 감정 다중분류 모델은 거의 없는 것일까?

Bert implementation for the Korean multi-class sentiment analysis

1. Install dependencies:
```
!pip install mxnet
!pip install gluonnlp pandas tqdm
!pip install sentencepiece
!pip install transformers==3.0.2
!pip install torch
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```

2. Populate data into `*.tsv` (we in this code assume 6-class classification tasks, based on Ekman's sentiment model)
3. Train (assuming gpu device is used, drop `device` otherwise)
4. Use (See below `# test` comment)
