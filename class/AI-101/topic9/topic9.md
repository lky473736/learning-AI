## learning-AI : AI-101
### topic 9 : 머신러닝, 딥러닝 기본 개념 정리 

- **임규연 (lky473736)**
- 2025.04.29.

------

### FAQ

- https://github.com/MyungKyuYi/AI-class/blob/main/README.md
- 중간고사, 기말고사에 출제됨
```
    인공지능 기초를 위한 FAQ

    1. 인공지능에서 지능에 해당하는 기능은 무엇인가?
    2. 인공지능의 종류 3가지에 대해서 설명하시오 (지도학습, 반지도학습, 강화학습)
    3. 전통적인 프로그래밍 방법과 인공지능 프로그램의 차이점은 무엇인가?
    4. 딥러닝과 머신러닝의 차이점은 무엇인가?
    5. Classification과 Regression의 주된 차이점은?
    6. 머신러닝에서 차원의 저주(curse of dimensionality)란?
    7. Dimensionality Reduction는 왜 필요한가?
    8. Ridge와 Lasso의 공통점과 차이점? (Regularization, 규제 , Scaling)
    9. Overfitting vs. Underfitting
    10. Feature Engineering과 Feature Selection의 차이점은?
    11. 전처리(Preprocessing)의 목적과 방법? (노이즈, 이상치, 결측치)
    12. EDA(Explorary Data Analysis)란? 데이터의 특성 파악(분포, 상관관계)
    13. 회귀에서 절편과 기울기가 의미하는 바는? 딥러닝과 어떻게 연관되는가?
    14. Activation function 함수를 사용하는 이유? Softmax, Sigmoid 함수의 차이는? 
    15. Forward propagation, Backward propagation이란?
    16. 손실함수란 무엇인가? 가장 많이 사용하는 손실함수 4가지 종류는?
    17. 옵티마이저(optimizer)란 무엇일까? 옵티마이저와 손실함수의 차이점은?
    18. 경사하강법 의미는? (확률적 경사하강법, 배치 경사하강법, 미치 배치경사하강법)
    19. 교차검증, K-fold 교차검증의 의미와 차이
    20. 하이퍼파라미터 튜닝이란 무엇인가?
    21. CNN의 합성곱의 역활은?
    22. CNN의 풀링층의 역활은?
    23. CNN의 Dense Layer의 역활은?
    24. CNN의 stride, filter의 역활? 필터의 가중치는 어떻게 결정되는가?
    25. RNN을 사용하는 이유와 한계점은?
    26. LSTM을 사용하는 이유와 한계점은?
    27. GRU을 사용하는 이유와 차별성은?
    28. 결정트리에서  불순도(Impurity) – 지니 계수(Gini Index)란 무엇인가?
    29. 앙상블이란 무엇인가?
    30. 부트 스트랩핑(bootstraping)이란 무엇인가?
    31. 배깅(Bagging)이란 무엇인가?
    32. 주성분 분석(PCA) 이란 무엇인가?
    33. Dense Layer란 무엇인가?    
```

<br>

### 중간고사 리뷰잉

#### (1) 전통적인 프로그래밍 vs 인공지능 프로그래밍
- 전통적인 프로그래밍 : 입력과 규칙을 동시에 넣어서 출력한다
- 인공지능 프로그래밍 : 입력과 정답을 동시에 넣어서 모델이 "규칙"을 출력시킨다

#### (2) 차원의 저주 (curse of dimensionality)
- 데이터의 feature가 많으면 multiple regression에서 최적의 plane은 곡면 상태일 것임 
- 데이터의 feature가 많다 -> 정보가 많다 -> 모델이 데이터를 이해하기 어려울 수 있다 
    - overfitting을 야기할 수 있음
- 해결책
    - (1) feature selection 
    - (2) PCA
    - (3) regularization

#### (3) 딥러닝 vs 머신러닝
- 머신러닝
    - 사람이 직접 data의 feature extraction을 하여야 함
    - 전문가적 지식이 필요함
- 딥러닝
    - 모델에 data만 집어넣으면 모델 자체에서 feature extraction과 학습을 수행

#### (4) underfitting vs overfitting
- underfitting
    - 학습 데이터를 충분히 학습하지 못하여 성능이 낮은 상황 (학습 데이터조차도 제대로 학습이 되지 않음)
    - 해결책
        - augmentation (증대)
        - 모델의 복잡도 높이기 (layer 추가, parameter 추가...)
- overfitting
    - 학습 데이터를 너무 잘 학습해서 새로운 데이터가 들어왔을 때 그에 대한 반응을 할 수 없다 (일반화적 문제가 생김, generalization error)
    - 해결책
        - 원시적 방법
            - 모델의 복잡도를 줄이기
            - 데이터를 더욱 늘리기
        - 모델 내의 방법
            - dropout
            - batch normalization
            - regularization (feature selection)

#### (5) activation function
- activation function을 쓰는 이유
    - layer가 선형적으로 추가되면 f(x)가 10개 있으면 10f(x)
    - 만약에 계수를 생략하였을 때 f(x), 그저 layer 하나만 있는 거랑 똑같음
        - layer의 비선형성을 추가하여 layer를 쌓는 것을 의미가 있게 함
- actication function의 종류
    - 계단함수
    - relu
        - f(x) = max(0, x)
        - ReLU는 양수면 입력값을 출력하고, 아니면 0을 출력한다. 양수의 입력값에 대해서는 기울기가 1이고, 음수 입력 값에 대해서는 기울기가 0이다
        - 근데 왼쪽이 0이라 gradient가 update되지 않는 문제
            - 그래서 f(x) = max (0.01x, x)와 같은 leaky relu 등장

    - sigmoid (0, 1)
        - 단점
            - **back propagation을 할 때 gradient kill되는 문제 (gradient vanishing problem)**
            - exponential base 때문에 오버헤드 발생 가능성
            - 속도가 느림

#### (6) cross-validation
- cross validation : train data 말고도 val data를 두어서 generalization error가 줄여지는지 확인하는 방법 (모델의 일반화 성능을 확인하자)
- K-fold cross validation : train을 K개의 fold로 나누고 랜덤하게 하나씩 선택하여 val set을 선택하고 나머지 N-1개는 train data로 두어서 학습을 진행하자

#### (7) hyperparameter tuning
- 사람이 직접 설정할 수 있는 파라미터는 하이퍼파라미터임
- 하이퍼파라미터의 조합을 통하여 모델의 성능이 되게 달라짐 -> 최적화하자 (batch size, epoch, optimizer...)
- 종류
    - grid search (brute force)
    - random search
    - 베이지안 최적화

#### (8) KNN

<img src="https://www.researchgate.net/publication/379526843/figure/fig1/AS:11431281233846941@1712155247029/Mechanism-of-KNN-Fig-1-shows-how-KNN-works-in-a-more-specific-way-the-green-circle.png" width="200px">

- 어떤 데이터포인트에 가장 가까운 K개의 데이터포인트들을 확인하였을 때, 그들의 레이블을 보고 다수결의 원칙으로 나의 데이터포인트의 클래스를 결정하는 방식
- 보통 K개는 홀수개임

#### (9) label의 불균형이 모델에 미치는 영향
- imbalanced-data가 포함되면 특정 클래스의 데이터만 많이 학습되어 거기에 모델이 편향될 가능성 있음
- 해결책   
    - SMOTE
    - augmentation...

#### (10) MSE
- 오차제곱법
    - $\sum(x_i-x')^2$
    - MSE는 cost function, regression에서 많이 사용됨. 오차를 줄이자

#### (11) decision tree - impurity (gini index)

#### (12) ensemble과 bootstraping
- ensemble : 모델을 여러개 섞어 쓰는 것
    - 예시 : random forest (decision tree를 여러 개 두어서 다수결의 법칙이나 평균을 내는 것)
- bootstraping : 샘플을 여러개 사용하는 것 (복원 추출)

#### (13) gradient의 의미와 경사하강법의 formula, 계산
- gradient : 기울기, 손실함수의 미분계수 
- 경사하강법의 수식 : $W_{t+1} := W_t - f'$

<br>

### FAQ 이외의 질문 (중요한 거)
#### classification과 regression의 차이
- 공통점 : 무언가를 예측한다
- 차이점
	- (1) classification : 이산 데이터
		-  0, 1, 2, 3... 이렇게 class하면 당연히 모델이 3을 우세하지 않을까? (모델에 영향을 미친다)
		- 그래서 연관성을 끊어주기 위해 one-hot encoding이 필요하다
	- (2) regression : 연속 데이터 (추세)
- 근데 이 두개가 같은 순간도 있다 
	- 만약에 이산 데이터의 간격이 매우 촘촘할 때면? 
		- classification과 regression과 다른게 없다

#### 경사하강법
- 목적 : 모델의 loss가 최소인 weight와 bias를 구하기 위하여 (실제값과 예측값의 오차를 최소화하기 위하여)
- $W_{t+1} = W_t - r * f'$
    - f' : 손실함수의 미분값 (기울기)
    - r : 학습율
- 종류
    - 배치경사하강법 : 모든 데이터를 이용하여 bias, weight를 구한다 -> 시간이 오래 걸린다 (특이성에 민감하지 않지만)
    - 확률적경사하강법 : 확률적 경사 하강법은 맨 처음에 random하게 starting data point를 잡아서 w_new = w_now * r + b 라는 식을 통해 global minima point를 찾는 최적화알고리즘이다. 매 스텝에서 한 샘플을 랜덤하게 가져와서 그에 대한 gradient를 계산하기 때문에 속도가 훨씬 빠르고 local minima를 탈출할 수 있지만, global minima를 찾을 것이라는 보장을 할 수가 없다.
    - 미니배치경사하강법 : 훈련 데이터를 미니배치라고 하는 작은 단위로 분할하여 (online) 학습하는 것으로, 이는 적은 데이터를 사용하기 때문에 한정된 자원에서 매우 유리하다. 장점으로는 GPU의 사용으로 행렬의 연산을 더욱 빠르게 할 수 있다는 것이다.
    - optimizer : adaboost, adam, rmsprop...

