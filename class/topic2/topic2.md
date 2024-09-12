## learning-AI : deep learning application (61357002)
### topic 2 : FAQ를 통한 머신러닝, 딥러닝 기본 개념 정리 2 

<br>

- **임규연 (lky473736)**
- 2024.09.12.

------

### FAQ
    - https://github.com/lky473736/learning-AI/blob/main/FAQ.md
    - https://github.com/MyungKyuYi/AI-class/blob/main/README.md
    - 중간고사, 기말고사에 출제됨

- **딥러닝과 인공지능의 차이점**
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

- **noise**
    - 만약에 48장의 별개의 사진이 있다면? (**Generative AI의 원리**)
        - 각 사진마다 feature을 압축함 -> 학습함
        - 학습 후 어떤 noise에 이전에 학습했던 사진의 feature을 추가하여 A를 만들어낼 수 있음 (딥페이크처럼)
        
<br>
        
- **self fine-tuning**
    - 성능 : 사용자의 피드백으로 이루어지는 fine-tuning < 스스로 하는 fine-tuning
    
<br>

- **mask**
    - 일부분이 가려진 데이터를 학습. 일부분을 복원하려는 방법
    - **Bidirectional Encoder Representations from Transformer(BERT)** : BERT, 문장의 특정 단어를 가려서 그 단어를 맞추는 self-supervised 방식 학습
    - https://velog.io/@hsbc/Masked-Modeling

- **데이터 질의 3대 요소 : 노이즈, 이상치, 결측치**
    - noise : f(x)에서부터 벗어나는 것
    - outlier : 아예 범위에서 벗어나는 것
    - null : 값이 뚫려있는 것
    
<br>

- **overfitting vs underfitting**
    - overfitting
        - outlier와 noise까지 학습함 || 모델이 복잡함 (차원이 높음)
        - 규제로 해결 (L1, L2) || 모델의 복잡성을 낮춤
        
    - underfitting
        - data의 records가 적을 때 || 모델이 덜 복잡할 때 발생
        - data의 records를 늘린다 (oversampling, SMOTE...)
        - feature engineering을 진행한다.
        - model의 복잡성을 늘린다.


- **cross-validation**
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

- hyperparameter
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

- **classification report에서의 수치값들**
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

