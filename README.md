# KoSentimentAnalysis

## Bert implementation for the Korean multi-class sentiment analysis
### 왜 한국어 감정 다중분류 모델은 거의 없는 것일까?에서 시작된 프로젝트

#### ARM64 환경에 최적화

0. Environment: Pytorch, Dataset: [감성 대화 말뭉치](https://aihub.or.kr/aidata/7978)
1. Install dependencies:
```
!pip install sentencepiece
!pip install torch
```

2. Populate data (we in this code assume 6-class classification tasks, based on Ekman's sentiment model)
3. Train
4. Validate & Use (See below `# test` comment)
