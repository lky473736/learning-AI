- **CNN(transfer-learning)_classification_diabetes(assignment)**
    - diabetes 데이터셋을 이용하여 이진 분류 base model을 만듦
    - base model을 이용하여 trainable을 조절하고, Dense layer를 추가하여 pre-trained 모델을 fine-tuning함.
    - 여기서는 diabetes 데이터셋을 fine-tuning할 때에도 사용

<br>

- **fine-tuning_classification_adult(assignment)**
    - diabetes 데이터셋을 이용하여 만든 이진분류 base model를 이용하여 adult 데이터셋에 적용한다.
    - ../../data/adult/adult_train.csv의 상관관계를 파악하여, 낮은 상관관계성을 보이는 feature은 제거하겠다. (그래서 diabetes 데이터셋 feature의 갯수랑 맞추기)