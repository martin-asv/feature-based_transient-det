import numpy as np
import tensorflow as tf
from scipy.signal import detrend
from sklearn.preprocessing import minmax_scale
from scipy.stats import truncnorm

"""
This file contains the Xue & Freymueller (2023) synthetic time series generator
It has been slightly modified and commented.
"""

# Random Seeds
tf.random.set_seed(42)
np.random.seed(42)

# Defines transient signal
def transient(n, size, duration, center):
    t = np.arange(0,n)
    d = size/(1 + np.exp(-(t - center)/(duration/10.0), dtype=np.float128))
    return d

# Defines seasonal signal
# an = annual
# sa = semi-annual
def seasonal(n, an, sa, dt=1/365.25):
    t = dt*np.arange(n)
    c1 = an[0] + (an[1] - an[0])*np.random.rand()
    c2 = sa[0] + (sa[1] - sa[0])*np.random.rand()
    d = c1*np.sin(2*np.pi*(t + np.random.rand())) + c2*np.sin(4*np.pi*(t + np.random.rand()))
    return d

# Defines colored noise
# wn = white noise
# fn = flicker noise
# rw = random walk
def colored_noise(n, wn, fn, rw, dt=1/365.25):
    c1 = wn[0] + (wn[1] - wn[0])*np.random.rand()
    c2 = fn[0] + (fn[1] - fn[0])*np.random.rand()
    c3 = rw[0] + (rw[1] - rw[0])*np.random.rand()
    d = c1*white_noise(n) + c2*flicker_noise(n,dt) + c3*random_walk(n,dt)
    return d

# Define gaussian noise
def white_noise(n):
    y = np.random.randn(n)
    return y

# Defines flicker noise
def flicker_noise(n,dt=1):
    if np.remainder(n,2): #Checkea imparidad (si es impar, hace n par)
        nt = n + 1
    else:
        nt = n
        
    x = np.random.randn(nt)
    X = np.fft.fft(x)
    
    nf = n/2 + 1
    k = np.arange(nf,dtype=np.int64)
    
    X = X[k]
    X = X/np.sqrt(k+1)

    X = np.concatenate((X,np.conj(X[-1:0:-1])))
    
    y = np.real(np.fft.ifft(X))
    y = y[:n]
    
    y = y - np.mean(y)
    y = y/np.std(y)
    return dt**0.25*y

# Defines random walk noise
def random_walk(n,dt=1):
    L = np.tril(np.ones((n,n)))
    w = np.random.randn(n)
    y = np.dot(L,w)  
    return dt**0.5*y

# Defines outliers
def outliers(n, max_size, prob):
    c = np.random.rand(n) < prob #Boolean mask
    d = c*max_size*(np.random.rand(n)-0.5)*2 #Centraliza y escala para outliers + y - 
    return d

# Defines function for checking signal to noise ratio
def snr(d_signal, d_noise):
    std_signal = np.std(d_signal)
    std_noise = np.std(d_noise)
    return 10*np.log10((std_signal/std_noise)**2)

# Defines noise types by stablishing coef. for seasonal, colored noise and outliers
def random_noise(nt, noise_type=0):
    if noise_type == 0: # Vertical
        d_seasonal = seasonal(nt, [4.5,6.9], [0.7,1.5])
        d_noise = colored_noise(nt, [2.0, 2.9], [2.3, 5.9], [0, 2.9])
        d_outliers = outliers(nt, 10, 0.05)
    else: # Horizontal
        d_seasonal = seasonal(nt, [1,2], [0.2,0.5])
        d_noise = colored_noise(nt, [0.4, 0.9], [0.3, 1.6], [0, 1.4])
        d_outliers = outliers(nt, 5, 0.05)
    return d_seasonal + d_noise + d_outliers

# Added recently, but needs a new scaling method (for nan)
# Define random addition of non-valid data
def random_nan(n, n_nan, n_gaps, min_l, max_l):
    x = np.zeros(n)

    # Adds random points of NaN
    idx_nan    = np.random.choice(n, n_nan, replace=False) # Random positions
    x[idx_nan] = np.NaN # Replace values

    # Adds random gaps 
    gap_length = np.random.randint(min_l, max_l) # Random length
    for _ in range(n_gaps):
        start_idx = np.random.randint(0, n)        # Start
        end_idx   = min(start_idx + gap_length, n) # Ends
        
    x[start_idx:end_idx] = np.NaN # Replace values
    
    return x

# Temporal addition for testing influence in duration prediction
def random_sample(mu=0.8, sigma=0.2, p_normal=0.6, seed=None):
    """
    Genera un valor normalizado en [0, 1] a partir de una mezcla entre:
    - una distribución normal truncada con media `mu` y desviación `sigma`
    - y una distribución uniforme en [0, 1]

    Parámetros:
    - mu: media de la normal truncada (idealmente entre 0.6 y 0.8)
    - sigma: desviación estándar (idealmente 0.1 a 0.2)
    - p_normal: proporción de la mezcla que proviene de la normal truncada
    - seed: para reproducibilidad (opcional)

    Retorna:
    - Un valor escalar en el rango [0, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    use_normal = np.random.rand() < p_normal
    if use_normal:
        a, b = (0 - mu) / sigma, (1 - mu) / sigma
        return truncnorm.rvs(a, b, loc=mu, scale=sigma)
    else:
        return np.random.rand()

# Defines the generation of synth data
def MakeDataset(nt, event_num, noise_type=0, label_snr_threshold=None, add_nan=None):
    label = np.zeros(nt) # Empty array for label
    
    # Nan option
    if add_nan is None:
        d_noise = random_noise(nt, noise_type) # Creates random noise
    else:
        d_noise = random_noise(nt, noise_type)
        
        # Añade NaN
        n_nan  = add_nan['n_nan']   # Number of nan points
        n_gaps = add_nan['n_gaps']  # Number of gaps
        min_l  = add_nan['min_l']   # Min length for gap
        max_l  = add_nan['max_l']   # Max length for gap
 
        d_noise += random_nan(nt, n_nan, n_gaps, min_l, max_l) # Adds nan to the noise
        
    # Event generation
    n_event = np.random.choice(event_num, p=[0.4,0.6]) # Randomized between [0,1]
    d_transient = 0
    
    if n_event==1: # Transient signal
        # Randomize size
        #size = -1*30*random_sample()
        size = -1*30*np.random.rand() # For both directionn use *np.random.choice([-1,1])
        # Randomize duration
        duration = 10 + 30*np.random.rand()
        #duration = 10 + 30*random_sample(mu = 0.2) # 5 + 25*rand
        # Randomize center of sigmoid
        center = np.random.uniform(30, nt - 30)
        # Adds the transient signal
        d_transient += transient(nt, size, duration, center)
        
        # Get label (label depends on SNR if SNR is activated)
        index_start = int(center - duration/2)
        index_stop = int(center + duration/2)

        # SNR deactivated
        if label_snr_threshold is None:
            label[index_start: index_stop] = 1
        # SNR activated
        else: # Checks SNR for labeling
            if snr(d_transient[index_start: index_stop], d_noise[index_start: index_stop]) > label_snr_threshold:
                label[index_start: index_stop] = 1

    return d_transient + d_noise, label
    
# Minmax scale data
def scale(data):
    return minmax_scale(detrend(data, axis=1), axis=1)
    #aux = scl.fit_transform(detrend(data, axis=1).T)
    #return aux

def scale_nan(data):
    # Don't use, I didn't use it and needs some fixes
    # Realiza un minmax compatible con valores NaN
    x = np.arange(len(data))
    m, b, r_val, p_val, std_err = linregress(x[valid_mask], comp[valid_mask])
    d_data = comp - (m*x + b)

    # Norm (MM)
    test_data = (d_data - d_data[valid_mask].min())/(d_data[valid_mask].max() - d_data[valid_mask].min())