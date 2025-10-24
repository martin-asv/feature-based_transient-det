import os
import numpy as np

""" 
Small and no-thesis (not necessary) related functions
"""

def CheckFolder(folder_path):
    """Check if a folder exists; if not, create it."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def Prob2Class(prob, thr=0.5):
    """Returns a binary prediction vector from probabilities."""
    clss = np.zeros_like(prob)
    clss[prob > thr] = 1
    return clss

def Rotate2Az(east, north, azimuth_deg):
    """Returns the horizontal components of a station rotated to the convergence direction."""
    theta = np.deg2rad(azimuth_deg)
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    rotated = R @ np.vstack((east, north))
    return rotated[0], rotated[1]  # along_conv, cross_conv

def Checker(df): 
    """Checks and bypass NaN within moving window."""
    if df['value'].isna().sum() <= len(df)/2: 
        # If the number of NaN is less than half of the window length, replace with the mean of the window                                                          
        df.fillna(df.mean(), inplace=True)
    if df['value'].isna().sum() > len(df)/2:
        # In the other case, replace with zero
        df.fillna(.0, inplace=True)
    return df

def Invalidate(ts, pad=3): 
    """ Creates a new mask by padding existing NaN's"""
    mask = np.isnan(ts)
    new_mask = np.full(mask.shape, False)

    # Invalidate surrounding points
    for i in range(len(mask)):
        if mask[i] == True:
            start = max(0, i - pad)
            end = min(len(mask), i + pad)
            new_mask[start:end] = True # Extend invalid data
    return new_mask