import os
import glob
import numpy as np
import pandas as pd
import random
from collections import Counter  

'''
    tool/recurrence.py
    
    lky473736, Gyuyeon Lim
    2024.09.21.
'''

def split_sequences(sequences, n_steps) :
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

def split_sequences_freq(sequences, n_steps) :
    X, y = list(), list()
    
    for i in range(len(sequences)) :
        end_ix = i + n_steps

        if end_ix > len(sequences) :
            break

        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[i:end_ix, -1]  # 이전에는 seq_y에 sequences[end_ix-1, -1]가 할당되었음
        
        # seq_y에서 가장 빈도가 높은 클래스 찾기
        most_frequent_class = Counter(seq_y).most_common(1)[0][0]  # most_common은 Counter 클래스의 멤버 함수, 빈도수 높은 클래스 찾아줌
        
        X.append(seq_x)
        y.append(most_frequent_class)

    return np.array(X), np.array(y)

def split_sequences_existence(sequences, n_steps, what) : # 여기서 what이라고 하는 parameter가 바로 존재 여부를 확인할 클래스
    X, y = list(), list()
    
    for i in range(len(sequences)) :
        end_ix = i + n_steps

        if end_ix > len(sequences) :
            break

        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[i:end_ix, -1]
        
        is_here = 0 # 일종의 토큰 역할, 특정 클래스 유무에 따라 변할 변수
        if np.any(seq_y == what) : # 특정 클래스가 있다면
            is_here = 1 
            
        else : # 없으면
            is_here = 0
        
        X.append(seq_x)
        y.append(is_here)

    return np.array(X), np.array(y)