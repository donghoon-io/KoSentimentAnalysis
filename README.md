# KoSentimentAnalysis

## Bert implementation for the Korean multi-class sentiment analysis
### 왜 한국어 감정 다중분류 모델은 거의 없는 것일까?에서 시작된 프로젝트

0. Dataset: [감성 대화 말뭉치](https://aihub.or.kr/aidata/7978)
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
4. Validate & Use (See below `# test` comment)
