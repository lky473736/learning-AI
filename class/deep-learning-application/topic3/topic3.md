## learning-AI : deep learning application (61357002)
### topic 3 : 순환 데이터를 이용한 CNN의 구성

<br>

- **임규연 (lky473736)**
- 2024.09.19.

------

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

<br>

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

<br>

### 

### ResNet

- **Reference**
    - https://meaningful96.github.io/deeplearning/skipconnection/

- layer을 여러개 더하기
- 하는 이유
    - gradient vanishing problem을 해결
    - 잔차를 이용하여 학습을 용이하게 만듦
- 이전 Layer의 정보를 직접적으로 Direct하게 이용하기 위해 이전 층의 입력(정보)를 연결
- **예시 : ResNet에서 skip connection (shortcut) 도입**
    - **plain model과 잔차를 도입한 Resnet의 안정성 비교**
        <img src="./plain_vs_resnet.png" width="500px">
- shortcut을 만드는 방법 = 특정 layer와 layer를 concatenate 진행
    - 예를 들어서 LSTM(1) -> LSTM(2) -> LSTM(3) -> ...
    - LSTM(1)와 LSTM(3)을 concatenate하면, **이 둘의 차이를 학습**

<br>
