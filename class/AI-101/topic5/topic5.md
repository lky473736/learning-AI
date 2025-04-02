## learning-AI : AI-101
### topic 5 : 머신러닝, 딥러닝 기본 개념 정리 

- **임규연 (lky473736)**
- 2025.04.01.

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

### SVM

<img src="https://blog.kakaocdn.net/dn/dyU2w7/btqNe5n7NPN/0KrOvyF6RM8jYOTHwOYIv0/img.png" width="500px">

- SVM은 이중 분류에 특화되어 있으면서, 각 class를 잘 분류 가능한 hyperplane을 찾는 알고리즘임
- SVM 원리
    - 결정 경계를 찾는데, 결정 경계 주변에 있는 support vector와의 거리인 margin을 최대화할 수 있는 plane을 찾는 것
    - 차원 (즉 feature)가 늘어날 수록 어려우며, 위는 2차원이니 선형함수로 hyperplane을 찾은 거지만, n차원이면 n-1차원의 함수를 사용하여 plane을 찾아야 함
- SVM 종류
    - 선형 SVM 
        - soft margin : 오차를 어느 정도 허용하여, margin 안에 instance가 들어가도 됨
        - hard margin : margin 안에 instance를 절대로 허용하지 않음
        - C값을 조정하여 soft margin or hard margin을 선정
            - C가 커지면 overfitting될 가능성 높아짐
            - C가 작아지면 경계가 흐려져 underfitting될 가능성 높아짐
    - 비선형 SVM 
        - 비선형적이다 -> 차원을 한 단계 늘려서 classification하면 됨 -> 그러면 mapping function 사용해야함 -> 계산량 많아짐 -> "kernel" 사용 (실질적으로 차원을 늘리지 않고, 차원을 늘리는 것처럼 계산해줌)
        - kernel의 종류
            - RBF
                - gamma 파라미터가 생김 (기존 C에 추가)
                    - 전제 : 비선형 SVM이기 때문에 결정 경계에 곡률이 존재할 것
                    - gamma가 커지면 곡률이 커지면서 overfitting될 가능성 높아짐
                    - 작으면 곡률이 작아지면서 underfitting될 가능성 높아짐
            - sigmoid
            - polynormial

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

### **classification report에서의 수치값들**
- **Precision (정밀도)**  
    - Precision은 True Positive (TP)를 True Positive와 False Positive (FP)의 합으로 나눈 값
    - Precision = TP / (TP + FP)

- **Recall (재현율)**  
    - Recall은 True Positive (TP)를 True Positive와 False Negative (FN)의 합으로 나눈 값
    - Recall = TP / (TP + FN)

- **F1-Score (F1 점수)**  
    - F1-Score는 Precision과 Recall의 조화 평균
    - F1-Score = 2 × (Precision × Recall) / (Precision + Recall)

- **Support (지원)**  
    - Support는 클래스의 실제 양성 샘플 수를 의미
    - Support = TP + FN
    
- **왜 F1-score을 사용하는가?**

    | 1000 | 10   |
    |------|------|
    | 10   | 10   |

    - 만약에 confusion matrix가 위와 같다면, 아래는 accuracy 측면에서 보았을 때는 0.98 정도로 매우 높다. 하지만 diagonal의 특정 component 갯수가 굉장히 많은 것을 보자면, 문제에 따라서 classification이 제대로 이루어지지 않았다고 판단할 수 있겠다.
    - 예를 들어서, 코로나19 양성 표본과 음성 표본을 class로 두고 classification하는 문제라면 F1-Score는 Precision과 Recall의 조화를 평가하므로, 특히 양성 및 음성 표본의 예측이 중요한 문제에서는 양쪽 클래스에 대한 균형 잡힌 성능 평가를 제공할 수 있다는 것이다.
    - 음성 표본을 제대로 분류하지 못하면 실제 상황에서 큰 문제가 발생할 것임

<br>