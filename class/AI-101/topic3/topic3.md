## learning-AI : AI-101
### topic 3 : 머신러닝, 딥러닝 기본 개념 정리 3

- **임규연 (lky473736)**
- 2025.03.18. 

------

### classification과 regression의 차이
- 공통점 : 무언가를 예측한다
- 차이점
	- (1) classification : 이산 데이터
		-  0, 1, 2, 3... 이렇게 class하면 당연히 모델이 3을 우세하지 않을까? (모델에 영향을 미친다)
		- 그래서 연관성을 끊어주기 위해 one-hot encoding이 필요하다
	- (2) regression : 연속 데이터 (추세)
- 근데 이 두개가 같은 순간도 있다 
	- 만약에 이산 데이터의 간격이 매우 촘촘할 때면? 
		- classification과 regression과 다른게 없다

### 샘플링과 편향
- https://lazymatlab.tistory.com/129
	- `상식적으로 훈련하는 데이터와 테스트하는 데이터에는 도미와 빙어가 골고루 섞여 있어야 합니다. 일반적으로 훈련 세트와 테스트 세트에 샘플이 골고루 섞여 있지 않으면 샘플링이 한쪽으로 치우쳤다는 의미로 샘플링 편향 (sampling bias)이라고 부릅니다. `
- major class와 minor class
	- major class : 특정 클래스의 records의 수가 많음
	- minor class : 특정 클래스의 records의 수가 적음
- sampling bias
	- major class에 대하여 모델 학습 정도가 기울어진다
	- 예를 들어서 sitting 데이터가 많으면 모델이 sitting에 최적화되게 학습되기 때문
	- 따라서 균일한 레이블 갯수가 필요할 것이다

### 스케일링
<img src="https://blog.kakaocdn.net/dn/d3YdaH/btrGXgnfXjF/tyARmyWADktlBUWiTQ8Oak/img.png">

- 과연 저 초록색이 오른쪽 집단일까?
	- 그게 아니라는 것. 사실 왼쪽 집단.
	- 왜냐하면 weight의 간격과 length의 간격이 다르기 때문에 저렇게 보이는 것
- z-score normalization이 필요

### AI 4단계
- data collection and EDA
- data preprocessing : 결측치, 노이즈, 이상치 처리, 스케일링, encoding...
- model fit : architecture의 정의, optimizer, loss function
- evaluation과 visualization