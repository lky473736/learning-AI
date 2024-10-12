'''
    Gyuyeon Lim (lky473736)
    augment_original.py
    
    Reference
    - https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data/blob/master/Example_DataAugmentation_TimeseriesData.ipynb
    - https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py
    - https://pypi.org/project/transform3d/
    - http://dmqm.korea.ac.kr/activity/seminar/390
    - https://hyeongyuu.github.io/machine%20learning/Augmentation_timeseries/
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline # for warping
from transforms3d.axangles import axangle2mat # for rotation

# Jittering: Adding noise to the data
def da_jitter(X, sigma=0.05) :
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + myNoise

# Scaling: Scaling the data
def da_scaling(X, sigma=0.1) :
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

# Magnitude Warping: Applying smoothly varying noise
def generate_random_curves(X, sigma=0.2, knot=4) :
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()

def da_mag_warp(X, sigma=0.2, knot=4) :
    return X * generate_random_curves(X, sigma, knot)

# Time Warping: Distorting the time steps
def distort_timesteps(X, sigma=0.2) :
    tt = generate_random_curves(X, sigma)
    tt_cum = np.cumsum(tt, axis=0)
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, 0], (X.shape[0] - 1) / tt_cum[-1, 1], (X.shape[0] - 1) / tt_cum[-1, 2]]
    tt_cum[:, 0] *= t_scale[0]
    tt_cum[:, 1] *= t_scale[1]
    tt_cum[:, 2] *= t_scale[2]
    return tt_cum

def da_time_warp(X, sigma=0.2) :
    tt_new = distort_timesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new

# Rotation: Rotating the data
def da_rotation(X) :
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X, axangle2mat(axis, angle))

# Permutation: Permuting segments of the data
def da_permutation(X, nPerm=4, minSegLength=100) :
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return X_new

# Random Sampling: Randomly sampling from the data
def rand_sample_timesteps(X, nSample=1000) :
    tt = np.zeros((nSample, X.shape[1]), dtype=int)
    tt[1:-1, 0] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[1:-1, 1] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[1:-1, 2] = np.sort(np.random.randint(1, X.shape[0] - 1, nSample - 2))
    tt[-1, :] = X.shape[0] - 1
    return tt

def da_rand_sampling(X, nSample=1000) :
    tt = rand_sample_timesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:, 0] = np.interp(np.arange(X.shape[0]), tt[:, 0], X[tt[:, 0], 0])
    X_new[:, 1] = np.interp(np.arange(X.shape[0]), tt[:, 1], X[tt[:, 1], 1])
    X_new[:, 2] = np.interp(np.arange(X.shape[0]), tt[:, 2], X[tt[:, 2], 2])
    return X_new

# All augmentation techniques applied to input data
def apply_augmentation(df, methods=None, n_samples=0, params=[]) :
    augmented_data = []

    for method in methods :
        match method :
            case 'jitter' :
                augmented_data.append(da_jitter(df.values, params[0])) # sigma
            case 'scaling' :
                augmented_data.append(da_scaling(df.values, params[0])) # sigma
            case 'magnitude_warp' :
                augmented_data.append(da_mag_warp(df.values, params[0], params[1])) # sigma, knot
            case 'time_warp' :
                augmented_data.append(da_time_warp(df.values, params[0])) # sigma
            case 'rotation' :
                augmented_data.append(da_rotation(df.values))
            case 'permutation' :
                augmented_data.append(da_permutation(df.values, params[0], params[1])) # nPerm, minSegLength
            case 'random_sampling' :
                augmented_data.append(da_rand_sampling(df.values, params[0]))

    print("##### augmentation complete #####")

    return pd.DataFrame(np.vstack(augmented_data))
