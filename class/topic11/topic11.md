## learning-AI : deep learning application (61357002)
### topic 11 : summary 및 앞으로의 방향성

<br>

- **임규연 (lky473736)**
- 2024.11.14.

------

- 시험 양식
    - ox로 낼 수도 있음 (ox를 많이 낼 수도 있음)
    - 문제 수는 아직 미정
    - 서술형 

------

- ICML, ICLR, CVPR, NIPS, AAAI -> 1년마다 트렌드가 바뀐다 -> 따라잡기 어렵다
- 논문을 열심히 보자 : 트렌드를 읽고 개념을 공부
- AI의 core concepts : 
- 딥러닝응용 이후에 해야할 일
    - 연구topic을 정하기 (vision, LLM, 시계열)
        - -> 직접 구현해보고 github, kaggle, dacon에 소스 찾아서 비교할 것

------

### Review

- (1) train, test로 나뉘는 이유
    - train data는 모델을 훈련하기 위해서, test data는 모델의 일반화 능력을 확인하기 위해서 
    - train data에 편중되면 overfitting
    - cross-validation하는 이유는?
        - train dataset을 폴드로 나누고 계속 번갈아가면서 일반화 성능 체크
- (2) self-supervised learning
    - 레이블을 스스로 만듦
    - 일종의 비지도학습 (https://sanghyu.tistory.com/1840)
- (3) 반지도학습 (semi-supervised)
    - 지도학습 + 반지도학습 (일부분은 label을 알려주어서 정확도를 올리는 방법)
    - 데이터를 섞거나 모델을 섞거나
- (4) masked learning을 하는 이유
    - 모델이 다양한 상황을 대처하게 하여 robust하게 모델을 만든다 
    - 일부분이 가려진 데이터를 학습. 일부분을 복원하려는 방법
        - Bidirectional Encoder Representations from Transformer(BERT): BERT, 문장의 특정 단어를 가려서 그 단어를 맞추는 self-supervised 방식 학습
        - https://velog.io/@hsbc/Masked-Modeling
- (5) feature selection vs feature engineering
    - feature selection : feature의 수가 너무 많으면 overfitting될 가능성이 높기 때문에 target과 상관관계가 높은 feature만 선택하여 학습 (성능이 더 좋아진다)
- (6) overfitting, underfitting 왜 발생하고, 해결법 무엇이고
    - overfitting 해결법
        - 모델파라미터를 줄이기
        - 규제를 추가
            - 모델 레이어에서도 normalization을 하는 이유
                - 모델 안에서도 데이터 자체의 분포가 변경될 수 있기 때문에
                - weight와 bias의 분포가 back propagation되면서 또 달라진다
        - data를 추가하여 다양성 추가
        - dropout, early stopping
    - underfitting
        - 모델파라미터 늘리기
        - 데이터 늘리기 (augmentation)
            - SMOTE (KNN)
        - 노이즈, 결측치 제거
- (7) confusion matrix, MSE, RMSE, MAE ...
    - confusion matrix : 혼동 행렬
        - 모델이 얼마나 헷갈려야하는지를 알 수 있음, 분류 모델의 예측 결과를 평가하는 데 사용되는 표
        - 쓰는 이유 : 어느 클래스에서 분류가 잘 안되는지를 알 수 있어서
    - F1 score을 쓰는 이유
        | 1000 | 10   |
        |------|------|
        | 10   | 10   |

        - 만약에 confusion matrix가 위와 같다면, 아래는 accuracy 측면에서 보았을 때는 0.98 정도로 매우 높다. 하지만 diagonal의 특정 component 갯수가 굉장히 많은 것을 보자면, 문제에 따라서 classification이 제대로 이루어지지 않았다고 판단할 수 있겠다.
        - 예를 들어서, 코로나19 양성 표본과 음성 표본을 class로 두고 classification하는 문제라면 F1-Score는 Precision과 Recall의 조화를 평가하므로, 특히 양성 및 음성 표본의 예측이 중요한 문제에서는 양쪽 클래스에 대한 균형 잡힌 성능 평가를 제공할 수 있다는 것이다.
        - 음성 표본을 제대로 분류하지 못하면 실제 상황에서 큰 문제가 발생할 것임
        - 이유 : **데이터 불균형을 고려하지 못한단 말임 -> 이걸 고려한게 F1**
    - precision을 쓰는 이유 & recall을 쓰는 이유 (https://mvje.tistory.com/200)
        - precision이 낮으려면 TP+FP가 낮아야함 
        - recall이 낮으려면 TP+FN이 낮아야함 
        - 각각의 상황마다 써야하는 지표가 당연히 다르다
- (8) embedding
    - 하는 이유 : ai가 각각을 이해하기 위해서 (숫자만 이해)
        - one-hot encoding하는 이유 : 각각의 편향성을 없애기 위해서 
        - vector encoding : {0.1, 0.3, 0.2, 0.7} 
        - encoding -> positional encoding
- (9) scaling을 하는 이유
- (10) ensemble
    - 모델을 섞어 쓴다
    - 샘플을 bootstraping한다 (중복을 허용하여 여러 개 샘플을 만듦 : 복원 추출)
        - OOB
    - boostraping-aggressive == 배깅
    - 페이스팅
- (11) hyperparameter tuning의 의미
    - hyperparameter : model을 구성할 때나 model을 동작시킬 때 사용자가 직접 조작할 수 있는 속성값
    - 이유 : 모델의 성능을 높이기 위해서, 최적의 성능을 가지는 모델을 만들기 위해서 (overfitting을 막기 위해서)
        - weight, bias는 ai 안에서 이루어지지만, ai를 모델링하는 것은 사용자가 함
        - 모델링을 어떻게 하는가에 따라 모델의 성능은 천차만별이다.
        - 최적의 조합이 무엇이냐를 찾는 것임
    - 종류
        - grid search : 브루트포스
        - random search : 균등분포
        - bayesian optimization
- (12) 경사 하강법
    - 목적 : 모델의 loss가 최소인 weight와 bias를 구하기 위하여 (실제값과 예측값의 오차를 최소화하기 위하여)
    - $W_{t+1} = W_t - r * f'$
        - f' : 손실함수의 미분값 (기울기)
        - r : 학습율
    - 종류
        - 배치경사하강법 : 모든 데이터를 이용하여 bias, weight를 구한다 -> 시간이 오래 걸린다 (특이성에 민감하지 않지만)
        - 확률적경사하강법 : 확률적 경사 하강법은 맨 처음에 random하게 starting data point를 잡아서 w_new = w_now * r + b 라는 식을 통해 global minima point를 찾는 최적화알고리즘이다. 매 스텝에서 한 샘플을 랜덤하게 가져와서 그에 대한 gradient를 계산하기 때문에 속도가 훨씬 빠르고 local minima를 탈출할 수 있지만, global minima를 찾을 것이라는 보장을 할 수가 없다.
        - 미니배치경사하강법 : 훈련 데이터를 미니배치라고 하는 작은 단위로 분할하여 (online) 학습하는 것으로, 이는 적은 데이터를 사용하기 때문에 한정된 자원에서 매우 유리하다. 장점으로는 GPU의 사용으로 행렬의 연산을 더욱 빠르게 할 수 있다는 것이다.
        - adaboost, adam, rmsprop...

- (13) SVM
    - hyperplane : 결정 경계
    - support vector의 의미
    - SVM은 margin이 최대화되어야 함

- (14) DT
    - 지니 불순도의 의미
    ```
    - **지니 계수(Gini Index)**: 지니 계수는 불순도를 측정하는 방법 중 하나입니다. 값이 0에 가까울수록 노드가 순수하며, 0.5에 가까울수록 불순합니다.
        - **수식**: $$ Gini = 1 - \sum_{i=1}^{n} p_i^2 $$, 여기서 p_i는 클래스 i의 비율입니다.
    ```
    - 불순도를 낮게 하는 것이 DT의 목적

- (15) 머신러닝에서 차원의 저주(curse of dimensionality)란?, dimensionality Reduction는 왜 필요한가?
- 차원의 저주 : 차원이 너무 많아질 수록 overfitting될 수도 있으며, 이해가 어렵다 (feature가 너무 많아서)
    - 이를테면 SVM에서 feature가 너무 많으면, hyperplane을 결정하기가 매우 까다롭고 직관적 이해가 거의 불가능해짐
        - 따라서 feature selection이나 feature engineering과 같은 기술로 이를 해결
        - 중요도가 높은 feature나 상관관계가 높은 feature만 selection하여 fit
        - 차원을 축소 (PCA)
        - **regularization**
- (16) PCA의 목적 : 정보를 압축시켜서 차원을 줄인다 (주성분을 찾아서)
- (17) manifold learning
    <img src='../topic2/manifold.png' width="500px">
    - 고차원에서는 결정 경계가 너무 명확하지 않다. 너무 복잡하다.
    - 고차원 데이터를 잘 하면 저차원으로 표현할 수 있다
    - **아무리 복잡한 데이터 (고차원) 이더라도 데이터 분석을 하여 특정 dimension으로 줄이면 파악하기 쉽다.**
    - 매니폴드 **가정** : 실제 고차원 데이터가 저차원 데이터에 가깝게 놓여 있다고 가정하는 것임
    - 예시 (차원 축소 알고리즘)
        - 스위스 롤 (2D manifold)
        - PCA
    - representation learning
- (18) activation function (활성화 함수)
    - layer 하나를 f(x)라고 할 때, layer를 n개 쌓으면 n * f(x), 상수는 필요 없음.
    - 그러면 입력과 출력이 거의 동일한 것 -> 의미 없음 -> 비선형적으로 만들어서 좀 더 복잡한 특징을 잘 추출하기 위함 (feature extraction을 더 잘하게 하기 위함)
    - 종류
        - sigmoid : 값을 0과 1 사이로 만들어줌 (출력층 : 이진 분류)
        - softmax : 출력층에서 
        - relu : x < 0에선 0, x > 0에서는 x 그대로
        - 계단 함수 단점 : 기울기 소실
        - ...
- (19) pre-trained model
   - 사전 학습의 장점, 특징
    - 이전 모델 (pre-trained model) 에서 학습된 가중치랑 bias를 전부 저장하는 것이기 때문에, 이전 모델을 다시 학습할 필요 없이 현재 문제를 해결할 수 있다 -> 시간이 절약된다
    - 이미 잘 만들어진 모델에다가 현재 domain에 맞는 layer를 추가하면 성능이 올라갈 것이다.
    - 작은 데이터셋에 대해 학습할 때 overfitting을 예방할 수 있다
        - 적은 데이터로 특징을 추출하기 위한 학습을 하게 되면, 데이터 수에 비해 모델의 가중치 수가 많을 수 있어 미세한 특징까지 모두 학습할 수 있음
        - 전이 학습을 이용해 마지막 레이어만 학습하게 한다면, 학습할 가중치 수가 줄어 과한 학습이 이루어지지 않게 할 수 있음.
    - pre-trained model에서의 input_shape와 현재 문제에서 사용할 데이터셋의 input_shape가 같아야 하기 때문에 reshape해주어야 함
    - freezing : 기존 모델의 가중치나 bias를 구할 필요가 없으니깐 함
    - fine-tuning : freezing 품, 값이 근삿값에 와있으니깐 빨리 학습 가능함

- (20) gradient vanishing, exploding
    - 각각 어떤 현상인지, 해결하기 위한 방법

- (21) CNN
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
        - pooling의 역할 두가지
            - 평행 이동 불변
            - 정보 요약
        - pooling의 장점 : 물체의 이동에 대하여 둔감하게 하는 것 

- (22)  ResNet
    - 목적
        - 잔차 학습을 도입하여 학습 용이하게 함
        - gradient vanishing problem 해결 위해
    - 잔차 학습 (skip connection)
        - gradient vanishing problem을 해결하기 위해 이전에 대한 정보를 반영한다 -> 입력과 이전것의 차이를 보아서 사전 정보가 있기 때문에 학습이 용이해진다
        - layer의 수를 많이 넣을 수 있다
    
- (23) inception
    - 이전과의 차이 : 서로 다른 filter를 학습한다
    - 여러 크기의 특성맵을 출력하는 합성곱 층을 구성
    - ```
        1x1 커널의 합성곱 층은 인셉션 모듈에서 중요한 역할을 합니다. 겉보기에는 한 번에 하나의 픽셀만 처리하기 때문에 공간적인 패턴을 잡지 못할 것처럼 보일 수 있지만, 실제로는 세 가지 주요 목적을 가지고 있습니다:

        1. **깊이 차원의 패턴 감지**: 1x1 합성곱층은 공간상의 패턴을 잡을 수 없지만, 채널(깊이) 차원을 따라 놓인 패턴을 잡을 수 있습니다. 즉, 채널 간 상관관계를 학습하는 데 유용합니다.

        2. **차원 축소**: 이 층은 입력보다 더 적은 특성 맵을 출력하므로, 병목층(bottleneck layer) 역할을 합니다. 차원을 줄이는 동시에 연산 비용과 파라미터 개수를 줄여 훈련 속도를 높이고, 과적합을 줄여 일반화 성능을 향상시키는 데 기여합니다.

        3. **복잡한 패턴 감지**: 1x1 합성곱층은 더 복잡한 패턴을 감지하기 위한 도구로 사용됩니다. 1x1 합성곱층과 3x3 또는 5x5 합성곱층의 쌍은 더 복잡한 패턴을 감지할 수 있는 하나의 강력한 합성곱층처럼 작동합니다. 이는 두 개의 층을 가진 신경망이 이미지를 훑는 것과 같은 방식으로, 더욱 정교한 특성을 학습할 수 있게 해줍니다.

        따라서, 1x1 합성곱층은 단순해 보일 수 있지만, 깊이 차원의 패턴을 감지하고, 차원을 축소하며, 더 복잡한 패턴을 감지하는 데 중요한 역할을 합니다.
        ```
    - 이전에 나온 inception 모듈이 팽이 모양

- (24) SENet
    - SENet : 채널에 대한 중요도 매기고 학습 용이 (기존 CNN에서의 채널의 중요도는 모두 동일하였다 -> SENet으로 중요도 매기고 중요한 채널만 온전히 학습하자 -> 그래서 light-weighted)
        - 채널 중요도 추출은 pooling을 이용
        - channel attention
        - **그러면 중요한 것만 집중하자 -> attention이 등장!**
    - Variant CNN의 목적
        - layer을 쌓으면서 gradient vanishing problem 해결
        - layer을 쌓으면 parameter 갯수가 많아지는데 이걸 light-weighted (경량화)
        
- (25) depthwise convolution
    - 채널별로 다른 필터를 사용하자
    - ```Inception-v4는 GoogLeNet과 ResNet의 아이디어를 결합하여 설계된 모델입니다. 이 모델에서 인셉션 모듈은 **깊이별 분리 합성곱층(depthwise separable convolution)**을 도입하여 연산 효율성을 극대화하고 성능을 향상시키는 데 기여합니다.

        **깊이별 분리 합성곱층(depthwise separable convolution)**이란, 일반 합성곱 연산을 두 단계로 나누는 방식을 의미합니다:
        1. **Depthwise Convolution**: 채널별로 개별적인 합성곱을 적용하여 공간 차원에서 패턴을 감지합니다.
        2. **Pointwise Convolution (1x1 Convolution)**: 1x1 합성곱을 사용하여 채널 간의 상호작용을 학습합니다.

        이 방식을 통해 **연산량을 줄이고** **모델의 효율성을 극대화**할 수 있습니다. 즉, 더 적은 계산으로도 복잡한 패턴을 감지할 수 있게 됩니다. 이러한 구조는 ResNet의 잔차 연결(residual connections)과 인셉션 모듈의 특징을 결합하여 **성능과 효율성**을 모두 강화시킨 모델입니다.

        Inception-v4에서도 이러한 분리 합성곱층을 사용하여 더 나은 성능을 달성하며, 동시에 모델의 복잡도를 조절하고 연산 비용을 줄일 수 있습니다.
        ```

- (26) RNN
    - RNN의 목적, LSTM의 목적, LSTM의 각각의 게이트의 역할, GRU 차이점

- (27) transformer
    - "Attention is all you need"
    - 모델은 중요한 부분에 더 집중하고, 덜 중요한 부분은 무시하여, 효율적으로 정보를 처리
    - 중요 요소
        - attention vs transformer 
            - attention은 정보가 sequence하게 입력됨
            - transformer는 한 뭉텅이가 하나로 입력됨 -> 순서가 없음

            - 예시) 예를 들어서 사과는 맛있는 과일이다 이러면
                - 사과는이라는 단어와 맛있는, 과일이다 이것의 상호관계에 대한 attention score를 계산하게 됨
                - 연관성이 얼마나 밀접한가를 중점
                - 각 단어마다 당연히 시간이 오래 걸림 -> 한꺼번에 병렬적으로 구하자 : multi-head attention

        - 임베딩 하는 이유
            - 모델에서 관계를 끊어주기 위함
            - 고차원 벡터 -> 저차원 벡터

        - positional embedding
            - 나 너 돈줘, 너 나 돈줘 <- 나, 너의 위치가 달라지기만 해도 의미가 달라짐
            - 위치에 따라 의미가 달라진다 -> 각 단어를 숫자로 변환하는 임베딩이 필요하며, 그 단어의 위치까지 기억해야하는 상황이 일어난 것임

        - transformer에서 중요한 4가지
            - embedding
            - positional embedding
            - multi-head attention
            - transformer

    - RNN과 LSTM의 입력과 출력이 **고정**되어 있어서 출력 sequence의 길이가 유동적이고 미리 알 수 없는 경우엔 적합하지 않음 (예를 들어서 GPT. 단어의 길이가 고정적이지 않음) -> **seq2seq**
        - decoder-encoder 구조로 되어있음

- (28) scaled dot-product attention
    - 이유 : attention score가 하나가 너무 커져서 scaling하기 위함
    - 유사도 점수를 scale로 조정한다

- (29) transformerr, AE, VAE, GAN
    - 