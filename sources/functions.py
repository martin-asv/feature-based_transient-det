"""
Thesis related functions (usually with a procedural way)
"""

#--- Libraries ---#

import pandas as pd
import numpy as np
import random
from tsfresh import defaults
from tsfresh import extract_features, select_features, defaults
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection.relevance import calculate_relevance_table
import matplotlib.pyplot as plt
from sklearn import tree
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import STL

import sys
sys.path.append("/home/martin/proyecto/sources/")

import toolbox as tb

#--- Functions start here ---#

def RelevancePlot(X, y, FDR=0.05):
    """
    X is a DataFrame with the features from extract_features of TSFRESH
    y is a Series with the target values for every sample in X (categorical) 
    FDR is False Detection Rate, it determines how strict is the rejection line. It influences the number of relevant features 
    
    This function returns the number of relevant, irrelevant and constant features. Also gives a plot of relevance.
    """  
    
    FDR_LEVEL = FDR
    # FDR_LEVEL = defaults.FDR_LEVEL
    df_pvalues = calculate_relevance_table(X, y, chunksize=1, fdr_level=FDR_LEVEL)

    # Visualize relevance
    HYPOTHESES_INDEPENDENT = defaults.HYPOTHESES_INDEPENDENT

    print("# Total \t", len(df_pvalues))
    print("# Relevant \t", (df_pvalues["relevant"] == True).sum())
    print("# Irrelevant \t", (df_pvalues["relevant"] == False).sum(),
          "( # Constant", (df_pvalues["type"] == "const").sum(), ")")

    # Estimate rejection line
    def calc_rejection_line(df_pvalues, hypothesis_independent, fdr_level):
        m = len(df_pvalues.loc[~(df_pvalues.type == "const")])
        K = list(range(1, m + 1))
    
        if hypothesis_independent:
            C = [1] * m
        else:
            C = [sum([1.0 / k for k in K])] * m

        return [fdr_level * k / m * 1.0 / c for k, c in zip(K, C)]  

    rejection_line = calc_rejection_line(df_pvalues, HYPOTHESES_INDEPENDENT, FDR_LEVEL)

    # Plotting of P-values
    fig, axs = plt.subplots(1, 2, figsize=(12,4), dpi=300, sharey=True, gridspec_kw={'wspace': 0.1})
    df_pvalues.index = pd.Series(range(0, len(df_pvalues.index)))

    df_pvalues.p_value.where(df_pvalues.relevant)\
        .plot(style=".", color='g', label="Relevant features", ax=axs[0])

    df_pvalues.p_value.where(~df_pvalues.relevant & (df_pvalues.type != "const"))\
        .plot(style=".", color='r', label="Irrelevant features", ax=axs[0])

    df_pvalues.p_value.fillna(1).where(df_pvalues.type == "const")\
        .plot(style=".", color='yellow', label="Constant features", ax=axs[0])

    axs[0].plot(rejection_line, '--k', label="Rejection line (FDR = " + str(FDR_LEVEL) + ")")
    axs[0].set_xlabel("Feature #", fontsize=12, fontweight='bold')
    axs[0].set_ylabel("P-value", fontsize=12, fontweight='bold')
    #axs[0].set_title("Mann-Whitney-U")
    #axs[0].legend(loc='upper left', frameon=True)

    # Zoomed plot
    last_rejected_index = (df_pvalues["relevant"] == True).sum() - 1
    margin = 20
    a = max(last_rejected_index - margin, 0)
    b = min(last_rejected_index + margin, len(df_pvalues) - 1)

    df_pvalues[a:b].p_value.where(df_pvalues[a:b].relevant)\
        .plot(style=".", color='g', label="Relevant features", ax=axs[1])
    
    df_pvalues[a:b].p_value.where(~df_pvalues[a:b].relevant)\
        .plot(style=".", color='r', label="Irrelevant features", ax=axs[1])
    
    axs[1].plot(np.arange(a, b), rejection_line[a:b], '--k', label="Rejection line (FDR = " + str(FDR_LEVEL) + ")")
    axs[1].set_xlabel("Feature #", fontsize=12, fontweight='bold')
    axs[1].set_ylabel("P-value", fontsize=12, fontweight='bold')
    #axs[1].set_title("Mann-Whitney-U")
    axs[1].legend(loc='upper left', frameon=True)
    plt.suptitle('Relevance plot', fontsize=14, fontweight='bold')

    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.6)

# (Still haven't used this)
def FeaturePlot(df, y):
    """
    Plots and save the behavior of each feature in the dataset, for each sample and respective target value.
    df is the DataFrame that have all the information of samples and features values.
    y is the Series that contains the target value for each sample.

    Returns only plots
    """
    
    # List all features
    lista = df.columns
    
    # Directory for mass generation of images
    directory = img + 'features/'
    tb.CheckFolder(directory)
    
    # Loop over all features
    for i in range(len(lista)):
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        
        ax.scatter(df.iloc[y.values].index,df[lista[i]].iloc[y.values], s=0.5, c='g')
        ax.scatter(df.iloc[~y.values].index,df[lista[i]].iloc[~y.values], s=0.5, c='b')
        plt.title(lista[i])
        plt.xlabel('Samples')
        plt.savefig(directory + lista[i] + '.png', dpi=300, transparent=True, format='png')
        plt.ioff()
        plt.close()

# (Modify this if there is a need of saving the plots)
def CostComplexityPruning(X_train, y_train, X_test, y_test):
    """
    Function to compare the performance and architecture of decision trees based on cost-complexity pruning.
    Returns the optimal alpha value and plots performance curves.

    X is a DataFrame product of extract_features of TSFRESH.
    y is a Series with the target values of the smaples in X.

    X, y must be separated for training and testing.
    
    The function returns comparison plots for different alpha values, and a suggested optimal alpha. 
    """
    
    # For selecting ccp_alphas evenly
    def select_evenly(arr, num_items):
        # Calculate the step size
        step = len(arr) / num_items

        # Use list comprehension to select items at evenly spaced indices
        selected_items = [arr[int(i * step)] for i in range(num_items)]

        return np.array(selected_items)
    
    # Initialize classifier and compute pruning path
    clf1 = tree.DecisionTreeClassifier(random_state=0)
    path = clf1.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    print('Number of alpha values:', len(ccp_alphas))
    
    # Limit the number of alphas to train on
    if len(ccp_alphas) > 50:
        ccp_alphas = select_evenly(ccp_alphas, 50)
        impurities = select_evenly(impurities, 50)

    # Plot impurity vs alpha
    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(ccp_alphas, impurities, marker="o", drawstyle="steps-post")
    plt.xlabel("Effective alpha")
    plt.ylabel("Total impurity of leaves")
    plt.title("Total Impurity vs effective alpha for training set")
    plt.show()

    # Initialize arrays for scores and node counts
    train_scores = np.zeros(len(ccp_alphas))
    test_scores = np.zeros(len(ccp_alphas))
    node_counts = np.zeros(len(ccp_alphas))
    depth = np.zeros(len(ccp_alphas))

    # --- Parallelization of alpha testing --- #
    def EvaluateAlpha(index, ccp_alpha):
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=abs(ccp_alpha))
        clf.fit(X_train, y_train)
        return (index,
            clf.score(X_train, y_train),
            clf.score(X_test, y_test),
            clf.tree_.node_count,
            clf.tree_.max_depth)

    # Run
    results = Parallel(n_jobs=-1)(delayed(EvaluateAlpha)(i, alpha) for i, alpha in enumerate(tqdm(ccp_alphas)))

    # Save results
    for i, train_acc, test_acc, nodes, tree_depth in results:
        train_scores[i] = train_acc
        test_scores[i] = test_acc
        node_counts[i] = nodes
        depth[i] = tree_depth

    # Plot number of nodes and depth vs alpha
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), dpi=300)
    ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
    ax[0].set_xlabel("Alpha")
    ax[0].set_ylabel("Number of nodes")
    ax[0].set_title("Number of nodes vs alpha")

    ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
    ax[1].set_xlabel("Alpha")
    ax[1].set_ylabel("Depth of tree")
    ax[1].set_title("Depth vs alpha")
    plt.tight_layout()
    plt.show()

    # Plot accuracy vs alpha
    plt.figure(figsize=(8,6), dpi=300)
    plt.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
    plt.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
    plt.xlabel("Alpha", fontsize=12, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12, fontweight='bold')
    plt.title("Accuracy vs alpha for training and testing sets", fontsize=14, fontweight='bold')
    plt.legend()
    plt.show()

    # Get optimal alpha
    best_alpha = ccp_alphas[np.argmax(test_scores)]
    print('Best alpha value: ' + str(best_alpha))

def NormalSelect(pos, SNR=None):
    """
    Uses a normal distribution from the positions of the time label to select a central point for the moving window when training.
    pos is an array with the positions where the transient signal is located.

    Returns a center position randomly selected from the normal distribution of all positions.

    Edit: Added option to make it SNR (signal to noise ratio) dependent, aiming to deal with duration overestimation.
    SNR (integer) is the threshold for modifying the std for the Normal distribution.
    """
    # Mean and standard deviation
    mu = np.mean(pos)
    if SNR==None:
        std = (pos[0] + pos[-1])/4  # Two if 1-sigma to the edge, 4 if 2-sigma
    else:
        std = (pos[0] + pos[-1])/8  

    # Normal distrib.
    normal_values = np.exp(-0.5 * ((pos - mu) / std) ** 2)
    normal_probs = normal_values / np.sum(normal_values)

    # Selection
    center = random.choices(pos, weights=normal_probs, k=1)[0]
    return center

# Keep if running without parallelization
def RunMovingWindow(ts, config, checker=False):
    """
    Classification over a moving window applied to a time series, with multiple models

    test_data: ndarray of shape (series, series_length)
    Config dictionary should have these variables defined:
        window: moving window length
        clf: trained classifiers
        clf_names: name of trained classifiers
        scaler: fitted sklearn StandardScaler
        settings: custom dictionary of tsfresh features (ej: ComprehensiveFCParameters())
        names: feature columns names (for reordering or filtering)
    verbose: show progress

    Returns
    padded_class: array of predicted classes for every model
    padded_prob : array of predicted probabilities 
    """
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)

    # Variables needed
    window    = config["window"]
    clf       = config["models"]
    clf_names = config["models_names"]
    scaler    = config["scaler"]
    settings  = config["settings"]
    names     = config["names"]
    
    len_series = len(ts)
    window_radius = window // 2
    it_range = range(window_radius, len_series - window_radius)

    # Prepare batch dataframe
    df_list = []
    for idx, center in enumerate(it_range):
        segment = ts[center - window_radius:center + window_radius]
        df = pd.DataFrame({'id': idx, 'time': np.arange(window), 'value': segment})

        # Added for non-valid data test
        if checker==True:
            df = tb.Checker(df)
        
        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)

    # Extract all features in batch
    features = extract_features(full_df, column_id='id', column_sort='time',
                                kind_to_fc_parameters=settings,
                                impute_function=impute, show_warnings=False,
                                disable_progressbar=False, n_jobs=8) # If multiprocessing doesn't work, get back to n_jobs=0
    features = features[names]
    features = scaler.transform(features)

    # Prepare result containers (full length, padded with NaNs)
    padded_class = np.full((len_series, len(clf_names)), np.nan)
    padded_prob  = np.full((len_series, len(clf_names)), np.nan)

    for idx in tqdm(range(len(features)), desc="Time series prediction"):
        for m_idx, model in enumerate(clf):
            y_pred = model.predict([features[idx]])[0]
            y_prob = model.predict_proba([features[idx]])[0][1]

            center = idx + window_radius
            padded_class[center, m_idx] = y_pred
            padded_prob[center, m_idx] = y_prob

    return padded_class, padded_prob

def RMW_MultipleSeries(series_array, config, checker=False):
    """
    Wraps RunMovingWindow() for multiple series and returns a DataFrame
  
    Check inputs for RunMovingWindow()

    Returns
    df_class : DataFrame with predictions for all time series
    df_prob  : DataFrame with probabilities for al time series
    """
    # Variables needed
    window    = config["window"]
    clf       = config["models"]
    clf_names = config["models_names"]
    scaler    = config["scaler"]
    settings  = config["settings"]
    names     = config["names"]
    
    n_series, len_series = series_array.shape
    all_class = []
    all_prob  = []

    for idx, ts in enumerate(series_array):
        print("___"*20)
        print(f"\n Processing serie {idx+1}/{n_series}")
        pred_class, pred_prob = RunMovingWindow(ts, config, checker)
        all_class.append(pred_class)
        all_prob.append(pred_prob)

    # Convert to arrays and reshape for DataFrame
    all_class = np.stack(all_class)  # shape: (n_series, len_series, n_models)
    all_prob  = np.stack(all_prob)

    id_time_index = pd.MultiIndex.from_product([range(n_series), range(len_series)], names=["id", "time"])

    df_class = pd.DataFrame(all_class.reshape(-1, len(clf_names)), columns=clf_names, index=id_time_index)
    df_prob  = pd.DataFrame(all_prob.reshape(-1, len(clf_names)), columns=clf_names, index=id_time_index)

    return df_class, df_prob

def ComposeDate(years, months=1, days=1, weeks=None, hours=None, minutes=None,
                 seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    """
    Takes time values (yeay, day, month ...) as input and returns a time array
    From Xue & Freymueller (2023), not modified.
    """
    years  = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days   = np.asarray(days) - 1

    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)

# Define function for estimating accumulated displacement
def EstimateDisplacement(pred, data, min_dur=10, verbose=True):
    """
    This function uses the prediction vector and data to fit and compute displacements during the detection window
    Detection windows are computed using a minimum duration to avoid noise or false detections.

    Input:
    pred   : Prediction vector for a station
    data   : Array wich contains the unscaled 3-component data for the station
    min_dur: Minimum duration imposed for the detection windows.

    Returns:
    valid, fit, disp, and error are dictionaries containing the values, duration, displacement 
    and error for every detection window computed
    """
    
    min_dur = min_dur # Minimum duration (shortest window possible for detection)

    # Allocate detections and initialize counter
    count = 0
    valid = []

    # Tracking of detection window (stard and finish)
    start = None
    for i, value in enumerate(pred): # Iterates over predictions
        # When prediction is 1, define the start
        if value == 1.0:
            if start is None:
                start = i 
                
        # When prediction value changes (to 0) and start is already defined, the window ends
        else:
            if start is not None:
                if (i - start) >= min_dur: # Evaluate window duration to save it
                    count += 1
                    valid.append((start, i - 1)) 
                start = None

    # Due to no more data, if there was a window, evaluate it
    if start is not None and len(pred) - start >= min_dur:
        count += 1
        valid.append((start, len(pred) - 1))

    # Report the count and range of the detections
    print("Number of valid detections for the station:", count)
    print("Valid detections range (start, finish):", valid)

    # Create dictionary to later access the fitted data, displacement and error
    fit, disp, error = {}, {}, {}

    # linear fit for very detection window
    for start, end in valid:
        # Extract the data needed
        t = np.arange(len(pred))[start:end]  # Detection time
        N = data[0, start:end]               # Data values
        E = data[1, start:end]
        
        # Linear fit
        Np, Ncov = np.polyfit(t, N, 1, cov=True)
        Ep, Ecov = np.polyfit(t, E, 1, cov=True)
    
        # Evaluate fit
        valN = np.polyval(Np, t)
        valE = np.polyval(Ep, t)
    
        # Save fitted data
        fit[(start, end)] = (valN, valE)
    
        # Estimate displacements
        dispN, dispE = round(valN[-1] - valN[0], 2), round(valE[-1] - valE[0], 2)
        disp[(start, end)] = (dispN, dispE)

        # Error elipse for displacement (from the fit error)
        Nsig = Ncov[0,0]*(end - start)
        Esig = Ecov[0,0]*(end - start)
        mag_sig = np.sqrt(Nsig**2 + Esig**2)
        error[(start, end)] = (Nsig, Esig, mag_sig) # Saves the propagated error and magnitude

        if verbose==True:
            print("___"*10)
            print(f"Window ({start}, {end}): North displ. = {dispN:.2f} [mm], East displ. = {dispE:.2f} [mm], Error mag. = {mag_sig:.2f} [mm]")
    
    # Valid contains the detections (and is used as index for fit, disp and error)
    # Fit contains the linear fit (array)
    # Disp contains the estimated displacement (scalar)
    # Error contains the propagated error in time of the linear fit, and the magnitude of the error
    return valid, fit, disp, error

def ReadTimeSeries(filename):
    """Reads data from files (SOPAC format), wraps ComposeDate()"""
    
    col_names = ['t', 'yr', 'doy', 'north', 'east', 'up', 'n_sig', 'e_sig', 'u_sig']
    df = pd.read_table(filename, sep=r'\s+', float_precision='round_trip', comment='#', names=col_names)
    
    # Convert 'yr' and 'doy' to datetime and replace 't'
    df['datetime'] = ComposeDate(df['yr'], days=df['doy'])
    
    # Drop unused columns for simplicity
    df = df[['datetime', 'north', 'east', 'up']]
    return df

def GetCommonRange(files, start_year=2005, end_year=2016):
    """Establish a common datetime range for all stations of the dataset."""
    
    # Read datetime indices from all files
    datetime_index = pd.DatetimeIndex([])
    for sta in files:
        df_timeseries  = ReadTimeSeries(sta)
        datetime_index = datetime_index.union(df_timeseries['datetime'])
    
    # Create a continuous time range (daily frequency)
    full_range = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-01-01', freq='D', inclusive='left')
    
    # Intersection ensures we stay within provided data range
    common_index = full_range.intersection(datetime_index)
    return common_index

# Main analysis pipeline
def ProcessStations(files, common_index):
    """
    Creates a matrix for all stations and components based on a common date range
    """
    n_t = len(common_index)
    all_sta = np.nan * np.zeros((len(files), 3, n_t))  # stations, components, time

    # Reads data for every file and reindex with common date range
    for i, sta in enumerate(files):
        df_timeseries = ReadTimeSeries(sta)
        
        # Reindex to make it continuous and align with common index
        df_timeseries = df_timeseries.set_index('datetime').reindex(common_index)
        
        # Populate the array with station components
        all_sta[i, 0, :] = df_timeseries['north'].values
        all_sta[i, 1, :] = df_timeseries['east'].values
        all_sta[i, 2, :] = df_timeseries['up'].values
    
    return all_sta

def STLImputation(tr):
    """
    tr must be a pandas Series object, optionally with a datetime index.
    Returns the index of datapoints imputed and the time series imputed.
    """
    # Make a copy of the original dataframe
    tr_cp = tr.copy()

    # Fill missing values in the time series
    idx = tr[tr.isnull()].index

    # Apply STL decompostion
    stl = STL(tr_cp.fillna(tr.mean()), seasonal=13, period=365)
    res = stl.fit()

    # Extract the seasonal and trend components
    sea = res.seasonal
    tre = res.trend

    # Create the deseasonalised (residual) series
    de_sea = tr_cp - sea - tre

    # Interpolate missing values in the deseasonalised (residual) series
    de_sea_imp = de_sea.interpolate(method="linear")

    # Add back the components
    tr_imp = de_sea_imp + sea + tre

    # Update the original dataframe with the imputed values
    tr_cp.loc[idx] = tr_imp[idx]

    return idx, tr_cp