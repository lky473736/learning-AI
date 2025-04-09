## learning-AI : AI-101
### topic 6 : 머신러닝, 딥러닝 기본 개념 정리 

- **임규연 (lky473736)**
- 2025.04.08.

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

### **딥러닝 필수 요소**
- loss function : https://github.com/MyungKyuYi/AI-class/blob/main/loss_function.jpg
    - 정답값과 예측값 사이의 오차를 계산함
    - regression : MSE, MAE, RMSE
    - classification : binary-crossentropy, categorical-crossentropy
- optimizer
    - SGD, Adam, RMSprop...
    - SGD
        - https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843
        - $W_{t+1} = W_t - r \times G$
            - r : learning rate
            - G : 현재 접선의 기울기
        - loss가 최소화되는 방향으로 weight와 bias를 찾는 최적화알고리즘
- activation function (활성화 함수)
    - layer 하나를 f(x)라고 할 때, layer를 n개 쌓으면 n * f(x), 상수는 필요 없음.
    - 그러면 입력과 출력이 거의 동일한 것 -> 의미 없음 -> 비선형적으로 만들어서 feature extraction을 더 잘하게 하기 위함
    - 종류
        - sigmoid : 값을 0과 1 사이로 만들어줌 (출력층 : 이진 분류)
        - softmax : 출력층에서 
        - relu : x < 0에선 0, x > 0에서는 x 그대로
        - ...
- back propagation
    - forward propagation의 결과와 실제 target과의 차이를 계산함
    - 차이를 최소화하는 방향으로 weight, bias를 찾아냄
    - gradient vanishing
        - 신경망이 깊어질수록 미분값이 0이 됨 -> RNN, LSTM
- forward propagation
    - 입력 -> 신경망 -> 출력
- one-hot encoding 
    - 각 label의 관계를 끊어주기 위해서 encoding
        - 1, 2, 3 이렇게 하면 1과 2가 가까워보이고, 1과 3이 멀어져 보임 
        - label을 공평하게 보기 위해서
    - label을 one-hot encoding해야 함 (categorical-crossentropy 시)

<br>

### 딥러닝할 때 해야하는 작업
- (1) 각 shape 확인
- (2) label은 one-hot encoding
    - label 사이의 연관성을 제거하기 위함
- (3) numpy로 바꾸기

<br>

### 경사하강법
- loss가 최소가 되는 weight와 bias를 찾는 최적화알고리즘
    - $W_{t+1} = r * W_{t} + b$
        - r : learning rate 
            - 학습율이 높다 : 빠르게 적응하지만 예전 데이터 금방 잊음
            - 학습율이 낮다 : 학습 속도가 느려지지만 노이즈에 덜 민감
        - b : bias
- 종류
    - 배치경사하강법 : 모든 데이터를 이용하여 bias와 weight를 구함
        - 장점 : 특이성에 민감하지 않다
        - 단점 : 시간이 오래 걸린다 + local minima + epoch 안에 global minima에 수렴하지 못할 가능성 있음
    - 확률적경사하강법 : 매 스텝에서 한 샘플을 랜덤으로 선택하고 그에 대한 weight와 bias를 구함
        - 장점 : 시간이 적게 걸림
        - 단점 : global minima에 수렴할 거라는 보장을 할 수 없음
    - 미니배치경사하강법 : 데이터셋을 일정한 크기의 미니배치 단위로 나누고, 한 배치를 택하여 학습을 진행, online 학습이 가능함
        - 장점 : 적은 데이터를 사용하기 때문에 한정된 자원에서 매우 유리함, GPU를 사용하여 빠른 병렬 연산이 가능해짐
