## learning-AI : AI-101
### topic 2 : 머신러닝, 딥러닝 기본 개념 정리 2 

<br>

- **임규연 (lky473736)**
- 2025.03.11. 

------

### 데이터셋을 불러오는 방법
- 컴퓨터에서 상대경로 혹은 절대경로 이용해서 파일 불러오기
- 라이브러리에 내장된 파일을 불러오기
- 웹에서 받아오기

<br>

### 복습

- 데이터셋을 불러오는 방법
    - 컴퓨터에서 상대경로 혹은 절대경로 이용해서 파일 불러오기
    - 라이브러리에 내장된 파일을 불러오기
    - 웹에서 받아오기
- X와 y로 나누는 이유 : X로부터 y를 알기 위해서 (일종의 함수처럼)
    - f(x) = y : X를 모델에 넣을 때 y가 나온다
    - ai에서의 모델의 수식은 f(x)말고 H(x)로 나타냄
- Train과 test로 나누는 이유
    - model의 성능을 측정. train 데이터로 학습하였는데 당연히 모델에 train 넣으면 정확도가 높을 거임
    - X_train, X_test, y_train, y_test
- 전통적인 프로그래밍과 인공지능 프로그래밍의 차이
    - 전통적인 프로그래밍 : 규칙, rule을 프로그래머가 프로그램에 직접 대입하여야 했음  
        - rule : 입력 데이터의 속성을 추출하는 것 (데이터의 확률분포를 파악하는 요소)
        - 알맞은 데이터 입력 -> 알맞은 target 출력함 (garbage in, garbage out)
    - AI 프로그래밍 : X와 y를 입력하면 rule을 생성한다 (이게 학습)
        - 위에서 feature가 X (sepal length, width, ...), label이 y (species)
        - 예측값과 실제값의 차이를 비교하여 acc, f1, precision, recall이라는 지표

<br>

### train과 test의 분포

- 만약에 train과 test의 분포나 속성이 다르다면?
    - 당연히 예측성능은 좋지 않을 것임
    - train과 test의 분포와 속성이 비슷할 거라고 **가정(hyperthesis)**
    - 많은 데이터셋이 필요한 이유 : generalization (일반화) 하기 위함

<br> 

### 학습의 의미 (기울기과 절편)

<img src = "https://miro.medium.com/v2/resize:fit:369/1*Ph5k6enitxYkBH-Cf0o5kQ.png">

<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FOoWt0%2FbtrhMSLXFbn%2FTmSJXVk1lVmENMEERlkBtK%2Fimg.png">

- 위는 도미 (파랑)와 방어 (주황)의 산점도
    - 위 산점도를 통해서 어떤 function을 볼 수 있다. (추세)
    - weight (W), bias (b)... 결국에 function이 polynormial이라면 아래와 같이 order에 따라 각각의 상수 및 계수를 구하는 것이 학습 
        - (일차식) H(x) = $ax + b$ (a와 b를 구해야 함)
        - (이차식) H(x) = $ax^2 + bx + c$ (a, b, c를 구해야 함)
        - **실제값과 예측값의 오차를 최소화하는 weight, bias를 구해야 한다**

<br>

### KNN

<img src="https://miro.medium.com/v2/resize:fit:591/1*kCqervQNQ5fGDfkFwrMzRQ.png">

- 지도학습의 한 일종 (K-means는 비지도학습)
    - 한 데이터포인트에서 가장 가까운 K개의 데이터포인트를 보고, 그 데이터포인트의 레이블을 확인하여 다수결 원칙을 통해 classification
    - 한 데이터포인트에서 가장 가까운 K개의 데이터포인트를 보고, 그 데이터포인트들의 가중평균을 계산하여 regression (https://blog.eunsukim.me/posts/how-to-regression-with-KNN)
- n_neighbors : 가장 가까운 K개의 데이터포인트
    - 최적의 K 결정 방법
        - elbow method (https://medium.com/@priyanshsoni761/k-nearest-neighbors-knn-1606989b7ee0)
        <img src="https://miro.medium.com/v2/resize:fit:980/1*uQpB2KYjcDaBBYoVZJ14Sw.png">
        - random search, grid search, K-fold

<br>

### 불균형 데이터

<img src="./countplot.png">

- 데이터의 레이블의 갯수를 확인한다
    - 위는 당뇨병 데이터, 당연히 당뇨병이 걸린 피험자 수보다 당뇨병이 걸리지 않은 피험자 수가 훨씬 많음
    - imbalanced data problem : major label data (수가 많은 데이터)에 더 많이 치우쳐져 학습될 경우가 있음
        - 해결 방법
            - train_test_split 시에 stratify 옵션 넣기
            - data augmentation

<br>

### **데이터 퀄리티 3대 요소 : 노이즈, 이상치, 결측치**
- noise : f(x)에서부터 벗어나는 것 (기존 분포에서 제거하는 것)
- outlier : 아예 범위에서 벗어나는 것
- null : 값이 뚫려있는 것

<br>

### pandas dataframe vs numpy ndarray
- pandas dataframe
    - 장점 : column명에 의해서 조작이 쉬움, manipulation 쉬움
    - 단점 : 느림
- numpy ndarray
    - 장점 : column이 없음. 하지만 속도가 빠름 (C 기반)
    - 단점 : column이 없음. manipulation 어려움
- tensorflow는 numpy로 변경해서 대입 / pytorch는 tensor로 변경해서 대입

<br>

### ML과 DL의 요소 차이

- ML과 DL에 데이터를 넣으려면 encoding이 필요 
    - one-hot encoding
    - label encoding
- ML의 label은 문자열이여도 됨

