## learning-AI : deep learning application (61357002)
### FAQ를 통한 머신러닝, 딥러닝 기본 개념 정리 

<br>

- **임규연 (lky473736)**
- **2024.09.12. 최종 수정**

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

### **인공지능에서 지능에 해당하는 기능은 무엇인가?**  
- 분류, 회귀, 인식 등을 기계가 하게끔 만들게 함 
    
<br>  

### **인공지능의 종류 3가지에 대해서 설명하시오 (지도학습, 비지도학습, 강화학습)**  
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

### **전통적인 프로그래밍 방법과 인공지능 프로그램의 차이점은 무엇인가?**  
- 전통적인 프로그래밍 : 규칙, rule을 프로그래머가 프로그램에 직접 대입하여야 했음  
    - rule : 입력 데이터의 속성을 추출하는 것 (데이터의 확률분포를 파악하는 요소)
    - 알맞은 데이터 입력 -> 알맞은 target 출력함 (garbage in, garbage out)
- AI 프로그래밍 : 데이터를 입력하면 rule을 자체적으로 생성  
    
<br>  

### **딥러닝과 머신러닝의 차이점은?**  
- CNN에서 feature extraction은 convolution layer에서 일어남 (모델 내에서 특성 추출 및 학습을 동시에 함)
- 딥러닝 : 특징 추출을 스스로 함 (feature extraction한 데이터를 대입하지 않음. 그대로의 원데이터를 넣음)
    - 하지만 딥러닝도 feature extraction을 한 데이터를 입력하면 정확도 상승
- 머신러닝 : 애초에 특징을 추출해서 대입하는 것 (correlation을 보고 feature을 선별하여 학습 ...)

<br>  

### **Classification과 Regression의 주된 차이점은?**  
- classification : 이산적 정보를 예측 (범주형 변수 / 분류)
- regression : 연속적 정보를 예측 (연속형 변수 / 회귀) + 추세를 발견 (weight, bias)
- **classification과 regression은 원리가 동일하다.**
    - classification의 class를 조금 더 세밀히 나누면 regression

<br>  

### **머신러닝에서 차원의 저주(curse of dimensionality)란?**, **Dimensionality Reduction는 왜 필요한가?** 
- 차원의 저주 : 차원이 너무 많아질 수록 overfitting될 수도 있으며, 이해가 어렵다
    - 이를테면 SVM에서 feature가 너무 많으면, hyperplane을 결정하기가 매우 까다롭고 직관적 이해가 거의 불가능해짐
        - 따라서 feature selection이나 feature engineering과 같은 기술로 이를 해결
        - 중요도가 높은 feature나 상관관계가 높은 feature만 selection하여 fit
        - 차원을 축소 (PCA)
        - **regularization**

<br>  

### **Ridge와 Lasso의 공통점과 차이점?**  
- ridge vs lasso : 규제 (regularization) -> overfitting 방지
    - ridge : L2, 제곱항이 결정계수 이하
    - lasso : L1, 절댓값 항이 결정계수 이하

<br>
    
### **Overfitting vs. Underfitting** 
<img src='https://media.geeksforgeeks.org/wp-content/cdn-uploads/20190523171258/overfitting_2.png' 
    width='500px'> 

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

### loss function
<img src='https://github.com/MyungKyuYi/AI-class/blob/main/loss_function.jpg?raw=true'>

- regression : MSE, MAE, RMSE...
- classification
    - 이진 분류 : binary_crossentropy
    - 다중 분류
        - one-hot encoding이 되어있지 않은 label : sparse_categorical_crossentropy
        - one-hot encoding이 되어있는 label : categorical_crossentropy

<br>

### **회귀에서 절편과 기울기가 의미하는 바는? 딥러닝과 어떻게 연관되는가?**  
- 절편과 기울기가 딥러닝에서 사용되는 것은 퍼셉트론 각각의 weight와 bias
- 딥러닝은 사실 weight와 bias를 epoch를 거듭하여 수정하여 최적의 parameter를 찾는 과정
- **SGD (Stochastic Gradient Descendant)**
    - https://github.com/lky473736/learning-AI/blob/main/insight/insight_2_SGD_concept_of_deep_learning.ipynb
    - https://towardsdatascience.com/step-by-step-tutorial-on-linear-regression-with-stochastic-gradient-descent-1d35b088a843

<br>

### **딥러닝과 인공지능의 차이점**
- 인공지능 : 많은 알고리즘 내포 (유전자 알고리즘, https://ko.wikipedia.org/wiki/%EC%9C%A0%EC%A0%84_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)
    - 규칙 기반 : 전문가 시스템, 초창기의 인공지능
        - rule을 계속 추가하면 한도 끝도 없다 -> 전문가 시스템이 망함
    - 통계 기반 
    - 신경 기반
- 머신러닝 : 특징 추출 및 학습
- 딥러닝 : 특징 추출과 학습을 동시에
    - **딥러닝은 입력과 출력 사이의 rule (f(x))를 구하는 것, 데이터의 특성을 발견해서 출력을 예측하는 것**
        - 방법 1. weight와 bias를 구하는 것
        - 방법 2. 확률적 모델 (데이터 분포 예측)
- U-net에서
    - latent space : 중요한 feature을 추출한 값들 (encoder, decoder 사이)
    - latent variable : 중요한 feature을 추출한 변수
    
<br>

### **noise**
- 만약에 48장의 별개의 사진이 있다면? (**Generative AI의 원리**)
    - 각 사진마다 feature을 압축함 -> 학습함
    - 학습 후 어떤 noise에 이전에 학습했던 사진의 feature을 추가하여 A를 만들어낼 수 있음 (딥페이크처럼)
    
<br>
    
### **self fine-tuning**
- 성능 : 사용자의 피드백으로 이루어지는 fine-tuning < 스스로 하는 fine-tuning

<br>

### **mask**
- 일부분이 가려진 데이터를 학습. 일부분을 복원하려는 방법
- **Bidirectional Encoder Representations from Transformer(BERT)** : BERT, 문장의 특정 단어를 가려서 그 단어를 맞추는 self-supervised 방식 학습
- https://velog.io/@hsbc/Masked-Modeling

<br>


### **데이터 퀄리티 3대 요소 : 노이즈, 이상치, 결측치**
- noise : f(x)에서부터 벗어나는 것
- outlier : 아예 범위에서 벗어나는 것
- null : 값이 뚫려있는 것

<br>

### **overfitting vs underfitting**
- overfitting
    - outlier와 noise까지 학습함 || 모델이 복잡함 (차원이 높음)
    - 규제로 해결 (L1, L2) || 모델의 복잡성을 낮춤
    
- underfitting
    - data의 records가 적을 때 || 모델이 덜 복잡할 때 발생
    - data의 records를 늘린다 (oversampling, SMOTE...)
    - feature engineering을 진행한다.
    - model의 복잡성을 늘린다.

<br>


### **cross-validation**
- 목적
    - 특정 세트에 편향된 기준으로 학습하지 않게끔 하기 위해
        - 만약에 데이터들이 다 균등분포면 굳이 할 필요 없다 (어차피 모든 데이터가 비슷한 분포인데 어떤 데이터에 편중되어 학습될 가능성이 0)
- 여러 개의 검증 세트를 사용한 반복적인 예비표본 검증 적용 기법 (**일반적인 모델을 만들기 위해**, 한쪽 데이터에서만 잘 작동하는 모델이 아니라 모든 데이터에 대하여 잘 작동하는 모델)
- 교차 검증 후 모든 모델의 평가를 평균하면 더 정확한 성능 측정 가능
— 교차 검증은 일반화 성능을 재기 위해 사용하는 훈련 데이터셋과 테스트 데이터 셋으로 한번 나누는 것보다 더 안정적이고 뛰어난 통계적 평가 방법
- 만약에 model.fit(..., validation_data = (X_test, y_test))이라는 소스가 있다면 이건 틀린 것임
    - validation_data가 test set이라면 test set에 집중하는 모델이 될 것임
    - 따라서 train set에서 validation set을 split하는 것임
    
<br>

### hyperparameter
- model을 구성할 때나 model을 동작시킬 때 사용자가 직접 조작할 수 있는 속성값
- ```<Defining Hyperparameters>
    Hyperparameters are the knobs and levers that machine learning engineers adjust before training. These parameters, which include the learning rate, batch size, number of epochs, and network architecture (e.g., number of layers and hidden units), remain fixed during training and significantly influence model performance.
    
    <Why Tune Hyperparameters?>
    The right hyperparameter settings can drastically improve a model’s ability to generalize from training data to unseen data. In contrast, poorly chosen hyperparameters may lead to overfitting or underfitting, thereby degrading the model’s performance on new tasks. Efficient hyperparameter tuning finds a sweet spot, balancing the model’s complexity and its learning capability.

    <Hyperparameter Types and Their Impact>
    Learning Rate: Dictates the adjustments made to model weights during training. Optimal settings ensure efficient convergence.
    Batch Size: Affects the model’s update frequency and convergence stability. Smaller sizes can enhance generalization but increase computation time.
    Network Architecture: More layers and units can model complex patterns but risk overfitting.
    Number of Epochs: Dictates how many times the learning algorithm will work through the entire training dataset. Too few epochs can result in an underfit model, whereas too many can lead to overfitting.
    Activation Functions: Functions like ReLU, sigmoid, and tanh impact the training dynamics and the model’s ability to approximate non-linear functions.

    <Basic Techniques>
    Grid Search: Exhaustive search over a specified parameter space.
    Random Search: More efficient than grid search, it tests a random combination of parameters.
    Bayesian Optimization: Utilizes a probabilistic model to predict the best hyperparameters.

    <Advanced Techniques>

    Automated Machine Learning (AutoML): Automates the selection and tuning of models, reducing the manual workload.
    Genetic Algorithms: Mimics natural selection to iteratively select the best model parameters.
    Hyperband: A resource-efficient method that speeds up the tuning process by focusing on promising parameter combinations.

    ```

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
    - <table border='1'><tr><td>1000</td><td>10</td></tr><tr><td>10</td><td>10</td></tr</table>
    - 만약에 confusion matrix가 위와 같다면, 아래는 accuracy 측면에서 보았을 때는 0.98 정도로 매우 높다. 하지만 diagonal의 특정 component 갯수가 굉장히 많은 것을 보자면, 문제에 따라서 classification이 제대로 이루어지지 않았다고 판단할 수 있겠다.
    - 예를 들어서, 코로나19 양성 표본과 음성 표본을 class로 두고 classification하는 문제라면 F1-Score는 Precision과 Recall의 조화를 평가하므로, 특히 양성 및 음성 표본의 예측이 중요한 문제에서는 양쪽 클래스에 대한 균형 잡힌 성능 평가를 제공할 수 있다는 것이다.
    - 음성 표본을 제대로 분류하지 못하면 실제 상황에서 큰 문제가 발생할 것임

