## learning-AI : deep learning application (61357002)
### topic 1 : FAQ를 통한 머신러닝, 딥러닝 기본 개념 정리 1

<br>

- **임규연 (lky473736)**
- 2024.09.05.

------

### FAQ
    - https://github.com/lky473736/learning-AI/blob/main/FAQ.md
    - https://github.com/MyungKyuYi/AI-class/blob/main/README.md
    - 중간고사, 기말고사에 출제됨

- **인공지능에서 지능에 해당하는 기능은 무엇인가?**  
    - 분류, 회귀, 인식 등을 기계가 하게끔 만들게 함 
    
<br>  

- **인공지능의 종류 3가지에 대해서 설명하시오 (지도학습, 비지도학습, 강화학습)**  
    - 지도학습 (supervised) : label이 존재  
        - 분류 : classification  
        - 회귀 : regression  
    - 비지도학습 (unsupervised) : label이 존재하지 않고, 패턴을 인식하여 clustering  
        - 요즘에 각광받음. <-- 현실에서 어떤 현상을 labeling하는 것이 쉽지 않으니깐
    - 강화학습 (reinforcement) : 사용자가 원하는 방향으로 보상을 주어 학습 (보상을 얻는 방향으로 학습)
        - 환경, 정책을 세우고 보상을 세움
        - LLM, GPT...
    - 반지도학습 : 지도학습 + 반지도학습 (일부분은 label을 알려주어서 정확도를 올리는 방법)
    - self-supervised : label이 없이 지도학습이 가능. 일종의 비지도학습 (https://sanghyu.tistory.com/1840)
    
<br>  

- **전통적인 프로그래밍 방법과 인공지능 프로그램의 차이점은 무엇인가?**  
    - 전통적인 프로그래밍 : 규칙, rule을 프로그래머가 프로그램에 직접 대입하여야 했음  
        - rule : 입력 데이터의 속성을 추출하는 것 (데이터의 확률분포를 파악하는 요소)
        - 알맞은 데이터 입력 -> 알맞은 target 출력함 (garbage in, garbage out)
    - AI 프로그래밍 : 데이터를 입력하면 rule을 자체적으로 생성  
    
<br>  

- **딥러닝과 머신러닝의 차이점은?**  
    - CNN에서 feature extraction은 convolution layer에서 일어남 (모델 내에서 특성 추출 및 학습을 동시에 함)
    - 딥러닝 : 특징 추출을 스스로 함 (feature extraction한 데이터를 대입하지 않음. 그대로의 원데이터를 넣음)
        - 하지만 딥러닝도 feature extraction을 한 데이터를 입력하면 정확도 상승
    - 머신러닝 : 애초에 특징을 추출해서 대입하는 것 (correlation을 보고 feature을 선별하여 학습 ...)

<br>  

- **Classification과 Regression의 주된 차이점은?**  
    - classification : 이산적 정보를 예측 (범주형 변수 / 분류)
    - regression : 연속적 정보를 예측 (연속형 변수 / 회귀) + 추세를 발견 (weight, bias)
    - **classification과 regression은 원리가 동일하다.**
        - classification의 class를 조금 더 세밀히 나누면 regression

<br>  

- **머신러닝에서 차원의 저주(curse of dimensionality)란?**, **Dimensionality Reduction는 왜 필요한가?** 
    - 차원의 저주 : 차원이 너무 많아질 수록 overfitting될 수도 있으며, 이해가 어렵다
        - 이를테면 SVM에서 feature가 너무 많으면, hyperplane을 결정하기가 매우 까다롭고 직관적 이해가 거의 불가능해짐
            - 따라서 feature selection이나 feature engineering과 같은 기술로 이를 해결
            - 중요도가 높은 feature나 상관관계가 높은 feature만 selection하여 fit
            - 차원을 축소 (PCA)
            - **regularization**

<br>  

- **Ridge와 Lasso의 공통점과 차이점?**  
    - ridge vs lasso : 규제 (regularization) -> overfitting 방지
        - ridge : L2, 제곱항이 결정계수 이하
        - lasso : L1, 절댓값 항이 결정계수 이하
        

<img src='https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171258/overfitting_2.png' 
     width='500px'>
- **Overfitting vs. Underfitting**  
    - https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/
    - overfitting : 모델이 train dataset에 너무 맞춰져서 새로운 데이터가 들어왔을 때 제대로 prediction을 하지 못함
        - **noise까지 학습 + outlier까지 학습** : 잘못된 학습 결과
        - 해결법 
            - model을 단순하게 만들기
            - **regularization : L1, L2**
    - underfitting : 모델이 train dataset을 제대로 학습하지 못함
        - 해결법 
            - model을 복잡하게 만들기
            - data augmentation

<br> 

- **딥러닝 필수 요소**
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

- **회귀에서 절편과 기울기가 의미하는 바는? 딥러닝과 어떻게 연관되는가?**  
    - 절편과 기울기가 딥러닝에서 사용되는 것은 퍼셉트론 각각의 weight와 bias
    - 딥러닝은 사실 weight와 bias를 epoch를 거듭하여 수정하여 최적의 parameter를 찾는 과정
    - SGD 참고

