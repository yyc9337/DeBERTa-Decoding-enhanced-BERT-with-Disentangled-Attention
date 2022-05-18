# DeBERTa-Decoding-enhanced-BERT-with-Disentangled-Attention
## 1. Introduction
Disentangled attention - 입력 계층의 각 단어 임베딩과 위치 임베딩의 합인 벡터를 사용하여 표현되는 BERT모델 과는 달리 DeBERTa의 단어는 각각의 내용과 위치를 인코딩하는 두 개의 벡터를 사용하여 표현되며 단어 사이의 어텐션 가중치는 이를 기반으로 분리된 행렬을 사용하여 계산된다. 이러한 방법은 단어 쌍의 어텐션 가중치가 내용뿐만 아니라 상대적인 위치에 따라서도 의미가 달라질 수 있다는 것을 의미한다.

Enhanced mask decoder - BERT모델과 마찬가지로 DeBERTa는 Masked language modeling(문장의 약 15퍼센트를 mask처리하고 mask처리된 문장을 예측하는)을 사용하여 사전 학습되는데, 기존에는 절대적 위치 정보에 영향을 받았지만 Relative Position Embedding이 전 계층에 영향을 주고 위치 정보를 보완 하기위해 Absolute Position Embedding을 추가하였다.

## 2. Background
Transformer - standard self-attention 메커니즘은 단어 위치 정보를 파악하는 것이 부족하여 입력 단어 임베딩에 위치 임베딩을 추가하여 위치 정보를 보완하였다.
Masked language model -  전체 시퀀스의 15% 마스킹하고 이를 복원한 예측값을 예측하도록 학습하였다.
## 3. DeBERTa Architecture
### 3.1 DISENTANGLED ATTENTION: A TWO-VECTOR APPROACH TO CONTENT AND POSITION EMBEDDING

내용(content)와 Position Embedding을 분리해서 접근합니다.





Content Embedding = H , Relative Position Embedding = P 

위의 식을 계산하여 HH, HP, PH, PP를 구하고  content-to-content, content-to-position, position-to-content, and position-to-position 표현된다. 단어 쌍의 어텐션 가중치가 content뿐만 아니라 상대적인 위치에 따라 달라지기 때문에 position-to-content 또한 중요하다고 말하지만, content-to-position, position-to-content 용어를 모두 사용하여 완전히 모델링한다. 따라서 postion-to-position은 많은 정보를 제공하지 않으므로 제거된다. 위 과정 후에는 QKV(Query, Key, Value)를 사용한 어텐션 연산 방식과 같습니다.

                                                     <Disentangeld Attention>


3.2 ENHANCED MASK DECODER ACCOUNTS FOR ABSOLUTE WORD POSITIONS

 DeBERTa는 상대적 위치 정보를 보완하기 위해 절대적인 위치를 고려했다. ‘쇼핑몰’(원래는 ‘mall’ 이라는 의미보다 ‘store’의 의미가 큰 것 처럼) 구문적 뉘앙스는 단어의 절대적인 위치에 크게 의존한다. DeBERTa에서는 모든 Transformer 계층 바로 뒤에 있지만 마스크된 토큰 예측을 위한 softmax 계층 바로 앞에 통합한다. 이러한 방식으로, DeBERTa는 모든 트랜스포머 레이어에서 상대 위치 정보를 보완하고 마스크된 단어를 디코딩할 때 보완 정보로 절대 위치만 사용한다. 
 DeBERTa의 디코딩 구성 요소를 강화 마스크 디코더(EMD)라고 부른다. 마지막으로 BERT가 사용하는 절대 위치의 통합 모델이 상대 위치의 충분한 정보를 학습하는 것을 방해할 수 있다고 추측하고 EMD를 사용하면 사전 교육을 위해 position이외에도 다른 유용한 정보를 얻을 수 있을 것이라 생각한다.

4. SCALE INVARIANT FINE-TUNING
 이 섹션에서는 fine-tuning을 위해 설명된 알고리즘의 변형인 새로운 가상 대립 훈련 알고리즘 Scale-invariant-Fine-Tuning(SiFT)를 제시한다.
