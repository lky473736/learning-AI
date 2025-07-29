### 일종의 회고 : 2025.07.28.

rehab은 rehabilitation의 줄임말로, '재활'이라는 뜻을 가지고 있습니다. 요즘에 드는 가장 큰 고민은 기록하는 습관이 없어졌다는 겁니다. 작년에 AI를 처음 배웠을 때는 코드를 통해 직접 구현하는 작업이 많았습니다. 멘땅에 헤딩이라는 말이 괜히 있는게 아닙니다. 예전엔 마크다운 파일과 주피터 노트북 파일을 총동원해서 제가 알고 있는 지식을 최대한 끌어다가 report를 작성했습니다. 굉장히 어려운 머신러닝 이론들을 저만의 언어로 쉽게 풀어서 써놓고는 했는데요. 

요즘은 이론 공부와 논문 작성에 집중하느라 논문 아이디어, 아키텍처 구현에 힘을 쏟아서 그런지 제 손으로 직접 아키텍처를 작성하는 것보다 claude나 다른 LLM에게 내 아이디어를 설명하고 코드를 대신 작성시키는 일이 대부분입니다. (물론 LLM이 완벽하게 코딩하는 건 절대 아니라서 수정은 내가 합니다.) 물론 제가 프롬프트질을 하고 있는 지금 작업이 어찌 보면 잘하는 것일수도 있습니다. 왜냐면 LLM은 더 거대해지고 강력해질테니 미리 이렇게 LLM에 친숙해지는 것도 좋겠죠. 

하지만 이게 너무 의존적이면 좋지 않다는 걸 많이 깨닿고 있는 지금입니다. 편리하고 실용적인 AI로 코딩을 하면서 나의 주피터 노트북에는 설명이 점점 줄어들었고, 어느새 아무 설명 없는 노트북만 즐비하게 되었습니다. data preprocessing하는 감각도 많이 잃어버린 것 같고, 확실히 컴퓨터공학을 하는 사람은 직접 코드를 쳐야 보람을 느끼는구나 싶습니다. 따라서 나는 오늘 재활을 시작하려고 합니다. 다시 한번 내가 아는 모든 걸 총동원하면서 직접 handcrafting해보려고 합니다. GPT의 도움을 단 하나도 받지 않고 오직 책과 논문에만 의존하는 고집 있는 작년의 저처럼 말입니다.

6~7월 동안 linear system과 파이토치 프레임워크 사용법을 공부했습니다. 따라서 파이토치 연습에도 매우 좋을테니, 여기 있는 모든 코드는 파이토치로 진행합니다. (물론 ML은 scikit-learn으로) 2025.07.28. 저녁 8시부터 다음 날 새벽 5시까지 (0), (1), (2), (3)을 진행하고, 2025.07.29. 오후 2시부터 밤 10시까지 나머지 것을 진행합니다. 그리고 앞으로 모든 주피터 노트북은 자세한 설명을 덧붙여서 진행할 겁니다.

- (0) diabetes (tabular) : 이미 했음. report/diabetes 참고
    - DNN 
- (1) number (sequence) : rehab_number.ipynb
    - split_seqeunces (sliding window) 
    - CNN-DNN
    - CNN-LSTM
- (2) EMG (sequence) : rehab_EMG.ipynb
    - CNN-LSTM (pure)
- (3) abalone (tabular) : rehab_abalone.ipynb
    - FCNN 
    - tabnet
- (4) california house price (tabular) : rehab_california_house_price.ipynb
    - FCNN
    - ML (KNN, LR, DT, RF, SVM, Logistic regression, XGB, Catboost, lightgbm)
- (5) Sisfall (sequence) : rehab_Sisfall.ipynb
    - autoencoder based ConvNet (FCNN)
    - latent space + transfer learning of FCNN
- (6) weather (sequence) : rehab_weather.ipynb
    - CNN-DNN
    - GRU
    - Residual Conn + Deep Conv

