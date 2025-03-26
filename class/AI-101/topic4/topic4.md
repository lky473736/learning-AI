## learning-AI : AI-101
### topic 4 :  머신러닝, 딥러닝 기본 개념 정리 

- **임규연 (lky473736)**
- 2025.03.25.

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
	- 왜?
		- (1) data가 부족
			- train data는 모델 훈련을 위해, test data는 모델의 일반화 성능을 확인하기 위해
				- 만약 train score가 val score보다 높으면서 gap이 크다 -> overfitting
		- (2) 모델이 너무 간단해서 그럼
    - 해결법 
        - model을 복잡하게 만들기
        - data augmentation (데이터를 늘리기)

### 왜 소스는 똑같은데 컴퓨터마다 성능이 다른가
- randomize 때문 (데이터 샘플링이 다를 수 있음)
	- **무작위성 (Randomness) 요소** :  일부 AI 모델은 실행할 때마다 랜덤성을 포함하는데, 같은 입력을 넣어도 매번 결과가 조금씩 달라질 수 있음
- 따라서 cross validation이 필요 (K-fold cross validation)
	- 하는 이유 : 데이터마다 특성이 각각 다르기 때문에 편향이 발생할 수 있으니 공정한 평가를 위해 
	- <img src="https://www.researchgate.net/publication/328798891/figure/fig1/AS:1050650980921344@1627506122858/Repeated-K-Fold-Cross-Validation-A-repeated-10-fold-CV-was-applied-The-10-fold-CV-works.ppm">
	- 보통 데이터를 3부분으로 나뉨
		- train : 학습에 쓰일 데이터셋
		- test : 평가에 필요할 데이터셋 (모델의 generalization)
		- val 

### normalization의 중요 (scaling)
<img src="https://velog.velcdn.com/images/ae__0519/post/a1554bd7-0074-4f55-a035-909dc053df7e/image.png">

- 위 세모는 어디에 포함되는가?
	- 잘 알 수 없다. 왜냐하면 weight와 height의 간격이 다르기 때문
	- scaling : 기준을 맞추기 위함
	- StandardScaler()를 통하여 z-score normalization을 할 수 있다

### KNR (K-Neighbors Regression)
- KNR : KNN의 Regression 버전임
	- 최근접 이웃 K개의 평균
	
### Classification과 Regression의 주된 차이점은?
- classification : 이산적 정보를 예측 (범주형 변수 / 분류)
- regression : 연속적 정보를 예측 (연속형 변수 / 회귀) + 추세를 발견 (weight, bias)
	- x와 y의 상관관계를 구한다 -> 그 상관관계를 구하기 위해서 결정계수라고 하는 metrics를 씀
- **classification과 regression은 원리가 동일하다.**
    - classification의 class를 조금 더 세밀히 나누면 regression

### 결정계수

<img src="https://blog-web.modulabs.co.kr/wp-content/uploads/2024/06/%EA%B2%B0%EC%A0%95%EA%B3%84%EC%88%98.jpg">

- 결정력이라고도 불리는 결정계수는 회귀분석의 성능 평가 척도 
	-   **독립변수가 종속변수를 얼마나 잘 설명하는 지**
	-   **0과 1 사이 값 (제곱을 했으니깐)** 을 가지고 상관계수가 높을 수록 1에 가까워지고 이는 모델의 설명력이 높음 
		- 제곱을 하는 이유 : -1\~1보다는 0\~1가 더 명확
	-   회귀 모델의 적합도를 평가할 때 사용됨

### 다항회귀
- 기존에 linear regression같은 경우에는 data의 추세를 잘 반영하는 거지만, 차원은 X와 y 두개임.
	- 차원을 늘렸을 때 데이터의 추세를 확인하고 싶으면 추세선은 추세면이 됨
	- <img src="https://miro.medium.com/v2/resize:fit:560/1*HLN6FxlXrzDtYN0KAlom4A.png">
	- **curse of dimensionality** -> 규제로 해결

### 다항 특성 만들기
- 새로운 특성을 만들어내거나 가공 == feature engineering 
- mean, std ... 이런 모수를 이용
- 새롭게 만들어낸 feature가 다른 feature와 상관관계가 낮았을 때 잘 만든거다
	- 비슷비슷하면 굳이 만들 필요가 없다


### regularization
- overfitting의 해결 방법 : 규제를 적용한다
    - layer 내에서 적용 가능
    - l2, l1 규제 : 불필요한 데이터의 특성을 줄이는 것
        - l2 : ridge : 제곱 규제 : feature을 가능하면 0에 가깝게 만드려고 함 (없애지는 않고 최소화하는 것임)
        - l1 : lasso : 절댓값 규제 : feature을 가능하면 0으로 만드려고 함 (사실은 이게 feature selection임. 일부 특성만 선택)

### logictic function

![](https://camo.githubusercontent.com/69aea9defb05f0039cb15ab4090fbf98c9f5e932565558fe954cc47e9dc178e6/68747470733a2f2f656469746f722e616e616c79746963737669646879612e636f6d2f75706c6f6164732f32333330326d61696e2d71696d672d37666339653836303163313565333339343537323038303061613233376137662e706e67)

-   발단
    
    -   (1) multiple regression에서 classification으로 변경 후에 범주형 y를 시각화하였더니, 중간 변수가 존재하지 않아 이산적인 그래프의 형태가 나타남
    -   (2) Y가 범주형 변수라면 다중선형회귀 모델을 그대로 적용할 수 없음
    -   (3) 그래서 logistic function (sigmoid function)을 도입하여 각 class에 대한 확률을 출력할 수 있도록 함
-   회귀식의 장점을 그대로 유지하고, 종속변수를 범주형이 아닌 확률로 둔 알고리즘
    
-   odds (승산)의 개념을 이용하여 함수를 휘게 만듦 (중간 값은 0.5)
    
    -   ![](https://camo.githubusercontent.com/f11e97bfe7ee5169b474f715d4fabd3e959c60281806d7669ab5063dd490a06f/68747470733a2f2f692e696d6775722e636f6d2f657577377151752e706e67)
    -   activation function
        -   이중 분류 : sigmoid
        -   다중 분류 : softmax

### DT (decision tree)

![](https://camo.githubusercontent.com/a87598dacb32127acfc5d1695371010665d675e57d2984431ea29359920e4f62/68747470733a2f2f626c6f672e6b616b616f63646e2e6e65742f646e2f4d384e76542f627471464e73637a5737532f374c375465634f43634539596c46494d4253774b4a302f696d672e706e67)

-   질문 (test)를 만들기 위한 학습을 진행하여, 각 class별로 효과적으로 분류하기 위한 모델이다.
- gini impurity (지니 불순도)
    - 현재 node에 들어온 각 component들의 class가 얼마나 섞여 있는지를 나타냄
    - 0.5에 가깝다 -> 불순도 낮음 / 0이다 -> 순수함 (한 클래스만 있음)
    - **결국에 불순도를 낮게 하는 것이 DT의 목적이 됨**
-   DT의 순서
    -   (1) 이등분의 효과가 가장 큰 test (gini가 가장 높은 것)가 가장 먼저 배치됨
        -   보통은 root node에 있는 test에 사용된 feature가 feature_importance가 가장 높은 경향이 있음
    -   (2) test를 계속 만들어 가지치기를 하고, 최종적으로는 leaf node가 순수히 한 class만 남도록 학습 (value에서 확인 가능)
-   DT의 장점
    -   시각적으로 매우 효과적으로 분류 이해 가능
    -   scaling을 굳이 하지 않아도 됨 (normalization)
-   DT의 단점
    -   DecisionTreeRegressor에서, 범위가 넘어간 값을 회귀로 예측 불가능 (오직 train set으로 학습한 범위 내에서만 예측이 가능하게 됨)

###  RF (random forest)

![](https://camo.githubusercontent.com/2d83987f6b11f0a3d6bc92540946078684ac51e383b8c38146e085f555685a79/68747470733a2f2f696d67312e6461756d63646e2e6e65742f7468756d622f523132383078302f3f73636f64653d6d746973746f72793226666e616d653d6874747073253341253246253246626c6f672e6b616b616f63646e2e6e6574253246646e2532465a4b344e36253246627471464e614a7a506a6725324673756b6f70444c44754b5349796d79324b575041596b253246696d672e706e67)

- ensemble : 다양한 모델을 동시에 돌려서 그 중 성능 좋은 걸 선택
    - 모델을 섞어 쓴다
    - 샘플을 bootstraping한다 (중복을 허용하여 여러 개 샘플을 만듦 : 복원 추출)
        - OOB
    - boostraping-aggressive == 배깅
    - 페이스팅
-   random으로 feature를 선정해 node로 지정 후 분기할 때마다 DT와 같은 방식으로 분기해 나가는 weaker learner를 만들어 비교 후, 다수결의 원칙에 따라 가장 많은 category로 예측
-   tree를 이용한 가장 확실하고 효과적인 방법, 일반적으로 decision tree보다 train score가 높다.
	- 장점 : 당연히 많이 돌리고 가장 좋은거 택하는거니깐 성능은 좋음
	- 단점 : overhead가 크다, 속도도 오래 걸림    
-   구성원리
    -   bootstrap sampling : 각각의 weaker learner를 만들기 위해 사용, feature을 복원 추출하고, 그 feature 안에서도 복원 추출하는 방법
	    - bootstrap : 동일한 샘플을 중복하여 여러개의 샘플을 만든다 (복원 추출)
    -   다음 분기를 고를 때에도 랜덤하게 뽑음
    -   OOB 검증 : 확실히 "복원 추출" 이기 때문에 봅히지 않은 샘플도 생기니, 이를 OOB에 넣어 validation test를 진행 (재활용성을 높인다)
 
