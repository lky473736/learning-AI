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

    | 1000 | 10   |
    |------|------|
    | 10   | 10   |

    - 만약에 confusion matrix가 위와 같다면, 아래는 accuracy 측면에서 보았을 때는 0.98 정도로 매우 높다. 하지만 diagonal의 특정 component 갯수가 굉장히 많은 것을 보자면, 문제에 따라서 classification이 제대로 이루어지지 않았다고 판단할 수 있겠다.
    - 예를 들어서, 코로나19 양성 표본과 음성 표본을 class로 두고 classification하는 문제라면 F1-Score는 Precision과 Recall의 조화를 평가하므로, 특히 양성 및 음성 표본의 예측이 중요한 문제에서는 양쪽 클래스에 대한 균형 잡힌 성능 평가를 제공할 수 있다는 것이다.
    - 음성 표본을 제대로 분류하지 못하면 실제 상황에서 큰 문제가 발생할 것임

<br>

### **일련의 tips**
- pd.read_csv에서 index_col=0을 하는 이유 : Unnamed :0가 가끔씩 나오기 때문에 이를 방지하기 위함
- 만약에 classification에서 label이 0-based가 아니거나, 불연속적일 때 : label-encoder로 0-based로 만들어주기
- 인덱스를 feature에 넣지 말아야 할 이유
    - id를 넣어봤자 차원만 늘어날 뿐임 (curse of dimensinality -> overfitting 야기)
- 파일을 읽어들였을 때 해야할 것
    - 이상이 있나 없나 (결측치, 이상치)
    - encoding
    - target countplot
- 모델 저장
    - save_model을 할 때 저장되는 값들 : weight, bias, 모델 파라미터...
    - 굳이 모델 저장하는걸 매번 할 필요 없음 (매번마다 다르니깐)
        - 하지만 **전이 학습**은 반드시 모델을 저장해놔야함
            - 전이 학습 : 신경망을 재사용하는 것
            - 재사용하는 이유 : 앞에서는 일반적인 특성을 추출하고, 뒤에서는 추상적인 특성을 추출하니깐 전이 학습 도입하여 편하게 함

<br>

### 차원의 저주
- feature의 갯수가 너무 많아서 모델이 데이터의 패턴을 파악하기 어려울 때
- 해결책    
    - **heatmap**을 구해서 상관계수를 보고 feature selection
        - 사실 feature의 조합을 모두 조합해서 학습해보고 점수를 비교해보면 된다.(wrapper)
            - 이러면 속도가 너무 저하되고 오래 걸림
        - **PCA** 를 이용한다
    - 궁극적으로 차원을 줄인다. (dimensionality reduction)

<br>

### manifold learning

![alt text](<스크린샷 2024-09-19 오전 9.29.58.png>)

- 고차원에서는 결정 경계가 너무 명확하지 않다. 너무 복잡하다.
- **아무리 복잡한 데이터 (고차원) 이더라도 데이터 분석을 하여 특정 dimension으로 줄이면 파악하기 쉽다.**
- 매니폴드 **가정** : 실제 고차원 데이터가 저차원 데이터에 가깝게 놓여 있다고 가정하는 것임
- 예시 (차원 축소 알고리즘)
    - 스위스 롤 (2D manifold)
    - PCA

<br>

### PCA

- https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-9-PCA-Principal-Components-Analysis
- train set에서 분산이 최대인 축을 찾기
- **t-SNE**

<br>

### 딥러닝 개요
- 필수 요소
    - optimizer
    - loss function
    - back propagation
    - forward propagation
    - one-hot encoding
- dense layer를 FCNN이라고도 한다 (full connected neural network)
- **모델의 복잡성을 늘린다는 것이 꼭 그렇게 좋은 것은 아님**
    - gradient vanishing problem
        - layer을 너무 깊게 쌓아서 미분하니 0에 수렴하는 문제 발생
        - 해결책
            - node수를 줄이거나, 모델을 작게 만들거나
            - **skip connection (shortcut) 도입**
                - layer을 여러개 더하기
                - 하는 이유
                    - gradient vanishing problem을 해결
                    - 잔차를 이용하여 학습을 용이하게 만듦
                - 이전 Layer의 정보를 직접적으로 Direct하게 이용하기 위해 이전 층의 입력(정보)를 연결
                - https://meaningful96.github.io/deeplearning/skipconnection/
                - **plain model과 잔차를 도입한 Resnet의 안정성 비교**
                ![alt text](<스크린샷 2024-09-19 오전 9.59.47.png>)

### 여러개의 입력, 여러개의 출력을 다루는 딥러닝

#### 입력이 여러개고, 출력이 1개
![ㅇㅇ](<스크린샷 2024-09-19 오전 9.46.49.png>)

- 입력이나 출력은 몇개도 될 수 있다. 
    - **단, 중간에 concatenate 이용하면 붙이기**
    - 위 소스를 확인해볼 때, feature을 겹쳐서 입력하였는데 이래도 아무 문제 없음

```python
input_wide = tf.keras.layers.Input(shape=[5])  # 특성 0 ~ 4
input_deep = tf.keras.layers.Input(shape=[6])  # 특성 2 ~ 7
norm_layer_wide = tf.keras.layers.Normalization()
norm_layer_deep = tf.keras.layers.Normalization()
norm_wide = norm_layer_wide(input_wide)
norm_deep = norm_layer_deep(input_deep)
hidden1 = tf.keras.layers.Dense(30, activation="relu")(norm_deep)
hidden2 = tf.keras.layers.Dense(30, activation="relu")(hidden1)
concat = tf.keras.layers.concatenate([norm_wide, hidden2])
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[input_wide, input_deep], outputs=[output])
```

#### 입력이 여러개고, 출력도 여러개

![](<스크린샷 2024-09-19 오전 10.05.32.png>)

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(loss=("mse", "mse"), loss_weights=(0.9, 0.1), optimizer=optimizer,
              metrics=["RootMeanSquaredError", "RootMeanSquaredError"])
    # 각 출력층마다 loss랑 weights가 다르다는 것에 주목
```

<br>

### CNN 기본

- CNN reference
    - https://www.freedium.cfd/https://towardsdatascience.com/deep-learning-illustrated-part-3-convolutional-neural-networks-96b900b0b9e0
    - https://www.freedium.cfd/https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b
    - https://www.freedium.cfd/https://medium.com/@koushikkushal95/understanding-convolutional-neural-networks-cnns-in-depth-d18e299bb438
    - https://www.freedium.cfd/https://medium.com/ai-mind-labs/convolutional-neural-networks-explained-a-hands-on-guide-7de893629686
- CNN을 사용하는 이유
    - 공간적인 정보 또한 학습시키는 것 (local feature을 추출하기 위함)
    - filtering을 위함 (feature extraction)
        - 처음에는 filter값이 랜덤 (**처음에 특정 filter을 넣을 수도 있음**)
        - 학습을 통하여 filter값을 최적화함 
    - 사용 목적
        - 이미지와 같이 부분적인 정보 추출을 할 때, DNN은 부적합 -> convolution 연산을 사용함
        - 이미지 부분에 필터를 곱해서 convolution 연산을 수행 -> 중요한 feature extraction -> regression/classification이 용이해짐
    - 전체 정보가 한 뉴런에 들어가면 굉장히 학습율이 떨어짐 -> 부분적인 학습으로 각 이미지를 영역으로 seperate하여 학습하는 것이 중요
- stride (stride이 적을 수록 구해지는 정보가 많아짐), padding, filter (1, 3, 5, 7, 9...)
- filter를 움직이는 stride
- filter값은 초기에 랜덤하게 정해짐 -> 데이터를 잘 설명할 수 있는 최적값으로 update됨 (back-propagation을 통하여 각 component가 결정되는 것)
- convolutional layer -> pooling 
    - 정보 압축 (parameter 줄이고, 성능 높이기 + 크기 줄이기)
    - pooling의 역할 
        - 평행 이동 불변
        - 정보 요약
    - pooling의 장점 : 물체의 이동에 대하여 둔감하게 하는 것 

<br>

### RNN 기본

- RNN Reference
    - https://www.freedium.cfd/https://medium.com/learn-love-ai/introduction-to-recurrent-neural-networks-rnns-43238d037a5c
    - https://www.freedium.cfd/https://towardsdatascience.com/deep-learning-illustrated-part-4-recurrent-neural-networks-d0121f27bc74
    - https://www.freedium.cfd/https://medium.com/ai-in-plain-english/lstm-architecture-in-simple-terms-491570fae6f0

- DNN, CNN은 데이터의 추세성을 반영한 것이 아니다. 이전의 시점을 반영하는 것도 아니다. 
- 전에 있었던 데이터를 통하여 다음 데이터를 예측하는 모델 (이전 시점을 반영)
    - 시계열 데이터, 주식 데이터, EMG, HARTH...
    - 어제 시장가를 확인한 후에 오늘 매각할지 매수할지를 결정하는 것처럼

- 순환 데이터
	- 순환 데이터를 고려할 때, regression인 경우엔 window 수 + 다음 수로 데이터를 만들면 됨.
	- classification이면 class의 갯수가 많은 것 or 맨 끝의 것
		- 0 1 1 1
		- 많은 것이 1, 끝의 것이 1

- RNN의 단점
	- 미분값의 손실 (기울기 소실) -> 문장이 길어지면 기울기 소실이 일어남
        - **LSTM의 등장**

<br>

- **LSTM**
![alt text](<스크린샷 2024-09-19 오전 10.53.20.png>)

    - reference
        - TinyML pdf file 

    - short term뿐만이 아닌 long term까지 학습
        - short term은 현재 학습하는 시계열의 시간 t
        - long term은 여태껏 학습해온 시계열들의 정보 (cell state / 중요한 정보만)

    - RNN은 layer가 많이 중첩될 수록 (cell이 많아질 수록) gradient vanishing 혹은 gradient exploding 문제가 발생한다.
        - 당연히 그럴만도 한게, 학습이 진행되면서 이전의 정보는 점차 희석되기 마련이다.
        - 따라서 가장 최근에 들어온 데이터셋에 대한 정보는 또렷히 학습된다.
            - -> **LSTM과 GRU가 도입된다.**
        - Long Short term memory : memory cell (cell state)의 도입으로 인하여 이전 문제점을 해결
        - LSTM의 구조 
            - (1) forget gate : 이전 hidden state의 일부를 까먹게 함 
                - 중요한 정보만 남긴다
                - sigmoid랑 연산하여 특정 확률을 cell state에 남길 것인가, 남기지 않을 것인가
            - (2) input gate : 새로운 정보가 cell state에 더해짐 (현재 입력 정보를 반영)
            - (3) output gate
                - 결과를 출력
                - 다음 cell에 hidden state, cell state를 넘겨줌
    - GRU 
        - LSTM과 비슷한데, gate를 하나 덜 사용한다. 하지만 LSTM과 비슷한 성능을 보이기에 효율적이다.

<br>

### DNN, CNN, RNN, LSTM 흐름 정리
    - DNN의 단점 : local feature인 공간적 특성을 추출하지 못한다
    - CNN의 단점 : 오직 현재 입력만 고려한다. (current state / 앞뒤 문맥을 파악할 수가 없다)
    - RNN의 단점 : 문장이 길어지면 gradient vanishing (깊어질수록 손실이 일어난다)
    - LSTM이 등장  

<br>

### 순환 데이터
- Reference
    - https://github.com/lky473736/learning-AI101/blob/main/insight/insight_3_split_sequence_and_CNN.ipynb
    - https://velog.io/@tobigsts1617/CNN-for-univariatemultivariatemulti-stepmultivariate-multi-step-TSF
    - https://qna.programmers.co.kr/questions/14992/%ED%8C%8C%EC%9D%B4%EC%8D%AC-lstm-%EC%8B%9C%EA%B3%84%EC%97%B4-%EC%98%88%EC%B8%A1-%ED%95%B4%EB%B4%A4%EB%8A%94%EB%8D%B0-%EC%A0%9C%EA%B0%80-%EC%9D%B4%ED%95%B4%ED%95%9C-%EA%B2%83%EC%9D%B4-%EB%A7%9E%EB%82%98%EC%9A%94
    - https://machinelearningmastery.com/
- **시계열 데이터도 CNN이 좋다 (split_sequence)**
    - ```python
        def split_sequences(sequences, n_steps):
            X, y = list(), list()
            for i in range(len(sequences)):
            # find the end of this pattern
                end_ix = i + n_steps
                # check if we are beyond the dataset
                if end_ix > len(sequences):
                    break
                # gather input and output parts of the pattern
                seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
                X.append(seq_x)
                y.append(seq_y)
            return np.array(X), np.array(y) 
- 순환 데이터를 만들어야 하는 이유
    - 만약 데이터가 x1, x2, x3, x4, x5가 있을 때, 만약 따로따로 입력하면 x2가 x1 등 나머지와 독립적이기 때문에 이전 것을 반영하는 것이 아니다.
    - 따라서 **x3를 넣으려면 x1, x2와 같이 넣어야 한다.** (그래야 이전 것을 반영하는 것이니)
    - **split_sequence**을 사용한다.
- 순환 데이터를 고려할 때, **regression인 경우엔 window 수 + 다음 수로 데이터를 만들면 됨.**
    - x3을 target으로 놓을 땐, x1, x2가 입력, x3이 feature
- **classification이면 class의 갯수가 많은 것 or 맨 끝의 것**
	- 0 1 1 1
	- 많은 것이 1, 끝의 것이 1
- **따라서 split할 수 있는 method는 정말 많다.**
    - 상황마다 다를 것
    - regression이냐, classification이냐에 따라 label이 많은 것으로 할 것인지, 아니면 끝쪽을 할 것인지 등등 다름

### NLP
- input -> text embedding -> LSTM ...
- embedding
    - encoding의 종류
        - 숫자 인코딩
        - 원핫 인코딩
        - **텍스트 임베딩**
    - https://aws.amazon.com/ko/what-is/embeddings-in-machine-learning/
    - text embedding을 layer로도 할 수도 있고, 아니면 딕셔너리로도 할 수도 있고
    - euclidean distance를 구해서 가까운  것으로 embedding
- contrast learning
