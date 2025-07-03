import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Tuple, List


def load_data(karl_path: str, senseboard_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(karl_path), pd.read_csv(senseboard_path)

def detect_TWA_change(df: pd.DataFrame, value_col: str = 'TWA') -> pd.DataFrame:
    data = df[value_col].values
    value_diffs = np.abs(data[1:] - data[:-1])
    switch_indices = np.where(value_diffs > 8)[0] + 1
    return df.loc[switch_indices, ['Lat', 'Lon', value_col, 'SecondsSince1970']]\
             .assign(index=switch_indices)
"""
def detect_TWA_change(
    df: pd.DataFrame,
    value_col: str = 'TWA',
    window: int = 5,
    threshold: float = 2
) -> pd.DataFrame:

    # Smooth TWA using rolling mean
    smoothed = df[value_col].rolling(window=window, center=True).mean()

    # Compute differences between consecutive smoothed values
    value_diffs = np.abs(smoothed.diff())

    # Find indices where the change exceeds the threshold
    switch_indices = np.where(value_diffs > threshold)[0]

    # Clean up: remove any out-of-bounds due to rolling window NaNs
    switch_indices = switch_indices[(switch_indices > 0) & (switch_indices < len(df))]

    return df.loc[switch_indices, ['Lat', 'Lon', value_col, 'SecondsSince1970']].assign(index=switch_indices)
"""

def detect_COG_change(df: pd.DataFrame, value_col: str = 'COG', threshold: float = 10.0) -> pd.DataFrame:
    """
    Détecte les points où le COG change brusquement de plus de `threshold` degrés.
    
    Args:
        df (pd.DataFrame): le DataFrame contenant les données.
        value_col (str): la colonne à analyser (par défaut 'COG').
        threshold (float): seuil en degrés pour détecter un changement significatif.

    Returns:
        pd.DataFrame: sous-ensemble du DataFrame aux points de rupture détectés.
    """
    data = df[value_col].values
    value_diffs = np.abs(data[1:] - data[:-1])
    
    # Corrige pour passage de 359° à 0°
    value_diffs = np.minimum(value_diffs, 360 - value_diffs)

    switch_indices = np.where(value_diffs > threshold)[0] + 1
    
    return df.loc[switch_indices, ['Lat', 'Lon', value_col, 'SecondsSince1970']]\
             .assign(index=switch_indices)

def detect_simple_sign_switch(df: pd.DataFrame, value_col: str = 'TWA') -> pd.DataFrame:
    data = df[value_col].values
    signs = np.sign(data)
    switch_indices = np.where(signs[1:] != signs[:-1])[0] + 1
    if len(df) > 0:
        switch_indices = np.concatenate(([0], switch_indices, [len(df) - 1]))
    return df.loc[switch_indices, ['Lat', 'Lon', value_col, 'SecondsSince1970']]\
             .assign(index=switch_indices)
"""
def detect_simple_sign_switch(
    df: pd.DataFrame,
    value_col: str = 'TWA',
    window: int = 5
) -> pd.DataFrame:
    # Smooth the data with a rolling mean
    smoothed = df[value_col].rolling(window=window, center=True).mean()

    # Remove NaNs caused by rolling
    smoothed = smoothed.dropna()
    valid_indices = smoothed.index

    # Detect sign switches on smoothed data
    signs = np.sign(smoothed.values)  # Utiliser .values pour travailler avec des array numpy
    switch_positions = np.where(np.diff(signs) != 0)[0] + 1
    
    # Get the actual indices from the DataFrame
    switch_indices = valid_indices[switch_positions]

    # Add first and last if needed
    if len(valid_indices) > 0:
        first_idx = valid_indices[0]
        last_idx = valid_indices[-1]
        switch_indices = np.concatenate(([first_idx], switch_indices, [last_idx]))

    # Ensure matching index for result
    result = df.loc[switch_indices, ['Lat', 'Lon', value_col, 'SecondsSince1970']].copy()
    result['index'] = switch_indices
    return result
"""


def find_best_ranges(df: pd.DataFrame, index_col: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    indices = sorted(df[index_col].tolist())
    ranges = list(combinations(indices, 2))
    best_pair = max(
        ((r1, r2) for r1, r2 in combinations(ranges, 2) if r1[1] < r2[0] or r2[1] < r1[0]),
        key=lambda pair: (pair[0][1] - pair[0][0]) + (pair[1][1] - pair[1][0]),
        default=((None, None), (None, None))
    )
    return best_pair


def trim_ranges_to_non_overlap(r1: Tuple[int, int], r2: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    a1, a2 = r1
    b1, b2 = r2
    if a2 <= b1 or b2 <= a1:
        return r1, r2
    if a1 <= b1:
        return (a1, min(a2, b1)), (max(a2, b1), b2)
    else:
        return (b1, min(b2, a1)), (max(b2, a1), a2)


def max_successive_gap_in_range(df: pd.DataFrame, index_col: str, range_: Tuple[int, int]) -> Tuple[int, int, int]:
    mask = (df[index_col] >= range_[0]) & (df[index_col] <= range_[1])
    indices = sorted(df.loc[mask, index_col].tolist())
    max_gap = 0
    max_pair = (None, None)
    for i in range(1, len(indices)):
        gap = indices[i] - indices[i - 1]
        if gap > max_gap:
            max_gap = gap
            max_pair = (indices[i - 1], indices[i])
    return max_pair[0], max_pair[1], max_gap


def plot_trajectories(karl_df, senseboard_df, result_sb, result_kl, result_sb_man, result_kl_man):
    plt.figure(figsize=(10, 8))
    colors = {'SenseBoard': 'blue', 'Karl': 'green'}
    plt.scatter(senseboard_df['Lon'], senseboard_df['Lat'], c=colors['SenseBoard'], marker='x', s=5, label='Traj. SenseBoard')
    plt.scatter(karl_df['Lon'], karl_df['Lat'], c=colors['Karl'], marker='x', s=5, label='Traj. Karl')
    plt.scatter(result_sb['Lon'], result_sb['Lat'], c='yellow', marker='o', s=40, label='Changes TWA SenseBoard')
    plt.scatter(result_kl['Lon'], result_kl['Lat'], c='yellow', marker='o', s=40, label='Changes TWA Karl')
    plt.scatter(result_sb_man['Lon'], result_sb_man['Lat'], c='red', marker='o', s=40, label='Manoeuvre SenseBoard')
    plt.scatter(result_kl_man['Lon'], result_kl_man['Lat'], c='red', marker='o', s=40, label='Manoeuvre Karl')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectoires et changements de TWA')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_max_gaps(karl_df, senseboard_df, gap1: Tuple[int, int], gap2: Tuple[int, int]):
    plt.figure(figsize=(10, 8))
    colors = {'SenseBoard': 'blue', 'Karl': 'green'}
    for start, end in [gap1, gap2]:
        plt.scatter(senseboard_df['Lon'][start:end], senseboard_df['Lat'][start:end], c=colors['SenseBoard'], marker='x', s=10)
        plt.scatter(karl_df['Lon'][start:end], karl_df['Lat'][start:end], c=colors['Karl'], marker='x', s=10)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectoires dans les zones de plus grand écart')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
"""
def reduce_to_one_per_second(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['SecondsSince1970'] = df['SecondsSince1970'].astype(int)
    df = df.drop_duplicates(subset='SecondsSince1970')
    return df.sort_values('SecondsSince1970').reset_index(drop=True)
"""
def analyze_session(karl_path: str, senseboard_path: str):
    karl_df, senseboard_df = load_data(karl_path, senseboard_path)
    #karl_df = reduce_to_one_per_second(karl_df)
    #senseboard_df = reduce_to_one_per_second(senseboard_df)

    window = 20
    result_sb = detect_COG_change(senseboard_df)
    result_kl = detect_COG_change(karl_df)
    result_sb_man = detect_simple_sign_switch(senseboard_df)
    result_kl_man = detect_simple_sign_switch(karl_df)

    result_all = pd.concat([result_sb_man, result_sb, result_kl_man, result_kl], ignore_index=True)

    r1_kl, r2_kl = find_best_ranges(result_kl_man, "index")
    final_r1, final_r2 = trim_ranges_to_non_overlap(r1_kl, r2_kl)

    gap1 = max_successive_gap_in_range(result_all, "index", final_r1)
    gap2 = max_successive_gap_in_range(result_all, "index", final_r2)

    print("Max gap in first range:", [gap1[0], gap1[1]])
    print("Max gap in second range:", [gap2[0], gap2[1]])

    plot_trajectories(karl_df, senseboard_df, result_sb, result_kl, result_sb_man, result_kl_man)
    plot_max_gaps(karl_df, senseboard_df, (gap1[0], gap1[1]), (gap2[0], gap2[1]))
