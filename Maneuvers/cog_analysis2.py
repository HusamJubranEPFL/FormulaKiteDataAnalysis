import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal

def extract_boat_name(file_path: str) -> str:
    """
    Extract boat name from CSV file name (without extension)
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def detect_COG_changes_rolling_mean(df: pd.DataFrame, value_col: str = 'COG',
                                    threshold: float = 15.0, window: int = 20) -> pd.DataFrame:
    """
    Detect changes in COG by comparing rolling means before and after each point.
    """
    cog = df[value_col].copy()
    n = len(cog)

    diffs = []
    for i in range(window, n - window):
        before = cog[i - window:i]
        after = cog[i + 1:i + 1 + window]
        
        mean_before = np.mean(before)
        mean_after = np.mean(after)

        # Handle circular difference
        delta = np.abs(mean_after - mean_before)
        delta = min(delta, 360 - delta)
        diffs.append(delta)
    diffs = np.array(diffs)

    change_indices = np.where(diffs > threshold)[0] + window  # shift due to slicing

    return df.loc[change_indices, ['Lat', 'Lon', value_col, 'SecondsSince1970']].assign(index=change_indices)

"""def plot_full_trajectories(
    boat1_df,
    boat2_df,
    boat1_changes,
    boat2_changes,
    boat1_name="boat1",
    boat2_name="boat2",
    show_changes=True
):
    colors = {boat1_name: 'green', boat2_name: 'blue'}

    plt.figure(figsize=(10, 8))

    # Trajectoires des bateaux
    plt.scatter(boat2_df['Lon'], boat2_df['Lat'], c=colors[boat2_name], marker='x', s=10, label=f'Trajectory {boat2_name}')
    plt.scatter(boat1_df['Lon'], boat1_df['Lat'], c=colors[boat1_name], marker='x', s=10, label=f'Trajectory {boat1_name}')

    # Points de changement de cap si demandé
    if show_changes:
        plt.scatter(boat2_changes['Lon'], boat2_changes['Lat'], c='red', s=40, label='COG Change Points')
        plt.scatter(boat1_changes['Lon'], boat1_changes['Lat'], c='red', s=40)

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectories' + (' with COG Change Points' if show_changes else ''))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""

def plot_trajectories(
    boat1_df: pd.DataFrame,
    boat1_changes: pd.DataFrame,
    boat2_df: pd.DataFrame = None,
    boat2_changes: pd.DataFrame = None,
    boat1_name: str = "boat1",
    boat2_name: str = "boat2",
    show_changes: bool = True
):
    """
    Trace les trajectoires d'un ou deux bateaux avec, en option, les points de changement de COG.
    """
    plt.figure(figsize=(10, 8))

    # Couleurs
    colors = {boat1_name: 'green', boat2_name: 'blue'}

    # Bateau 1
    plt.scatter(boat1_df['Lon'], boat1_df['Lat'], c=colors[boat1_name], marker='x', s=10, label=f'Trajectory {boat1_name}')
    if show_changes and boat1_changes is not None:
        plt.scatter(boat1_changes['Lon'], boat1_changes['Lat'], c='red', s=40, label=f'{boat1_name} COG Changes')

    # Bateau 2 si présent
    if boat2_df is not None:
        plt.scatter(boat2_df['Lon'], boat2_df['Lat'], c=colors[boat2_name], marker='x', s=10, label=f'Trajectory {boat2_name}')
        if show_changes and boat2_changes is not None:
            plt.scatter(boat2_changes['Lon'], boat2_changes['Lat'], c='orange', s=40, label=f'{boat2_name} COG Changes')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectory' + (' with COG Change Points' if show_changes else ''))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


"""
def compute_longest_intervals(
    boat1_changes: pd.DataFrame,
    boat2_changes: pd.DataFrame,
    boat1_df: pd.DataFrame,
    boat2_df: pd.DataFrame,
    top_n: int = 2,
    boat1_name: str = "boat1",
    boat2_name: str = "boat2",
    min_duration_sec: float = 30.0,
    sog_derivative_threshold: float = 0.2,
    smoothing_window: int = 300
) -> list[dict]:
    
    # Mark source of change points
    boat1_changes['source'] = boat1_name
    boat2_changes['source'] = boat2_name
    
    # Combine and sort all change points chronologically
    all_changes = pd.concat([boat1_changes, boat2_changes], ignore_index=True)
    all_changes_sorted = all_changes.sort_values('SecondsSince1970').reset_index(drop=True)
    
    # Preprocess SOG data with smoothing and derivative calculation
    for df in [boat1_df, boat2_df]:
        # Smooth SOG data to reduce noise
        df['SOG_smoothed'] = df['SOG'].rolling(
            window=smoothing_window,
            min_periods=1,
            center=True
        ).mean()
        
        # Calculate robust derivative using central differences
        df['SOG_derivative'] = (
            df['SOG_smoothed'].diff(2) / 
            df['SecondsSince1970'].diff(2).abs()
        ).abs()
    intervals = []
    
    for i in range(len(all_changes_sorted) - 1):
        start = all_changes_sorted.loc[i, 'SecondsSince1970']
        end = all_changes_sorted.loc[i + 1, 'SecondsSince1970']
        
        # Get data for this interval
        boat1_interval = boat1_df[(boat1_df['SecondsSince1970'] >= start) & 
                                (boat1_df['SecondsSince1970'] <= end)].copy()
        boat2_interval = boat2_df[(boat2_df['SecondsSince1970'] >= start) & 
                                (boat2_df['SecondsSince1970'] <= end)].copy()
        
        if boat1_interval.empty or boat2_interval.empty:
            continue
            
        # Merge the two boats' data by time
        merged = pd.merge_asof(
            boat1_interval.sort_values('SecondsSince1970'),
            boat2_interval.sort_values('SecondsSince1970'),
            on='SecondsSince1970',
            suffixes=('_boat1', '_boat2')
        )
        
        # Find periods where both boats have stable speed
        stable_mask = (
            (merged['SOG_derivative_boat1'] < sog_derivative_threshold) &
            (merged['SOG_derivative_boat2'] < sog_derivative_threshold))
        
        # Group consecutive stable periods
        stable_groups = (stable_mask != stable_mask.shift(1)).cumsum()
        
        for group_id, group_data in merged[stable_mask].groupby(stable_groups):
            group_start = group_data['SecondsSince1970'].min()
            group_end = group_data['SecondsSince1970'].max()
            group_duration = group_end - group_start
            
            if group_duration >= min_duration_sec:
                # Calculate additional metrics
                avg_sog1 = group_data['SOG_boat1'].mean()
                avg_sog2 = group_data['SOG_boat2'].mean()
                
                intervals.append({
                    'start_time': group_start,
                    'end_time': group_end,
                    'duration': group_duration,
                    'boat1_name': boat1_name,
                    'boat2_name': boat2_name,
                    'avg_SOG_boat1': avg_sog1,
                    'avg_SOG_boat2': avg_sog2,
                    'SOG_variation_boat1': group_data['SOG_boat1'].std(),
                    'SOG_variation_boat2': group_data['SOG_boat2'].std(),
                    'stability_score': 1/(1 + group_data[['SOG_derivative_boat1', 
                                                       'SOG_derivative_boat2']].mean().mean())
                })
    
    # Sort by duration and stability score
    return sorted(intervals, 
                 key=lambda x: (-x['duration'], -x['stability_score']))[:top_n]
"""

def compute_longest_intervals(
    boat1_changes: pd.DataFrame,
    boat2_changes: pd.DataFrame,
    boat1_df: pd.DataFrame,
    boat2_df: pd.DataFrame,
    top_n: int = 2,
    boat1_name: str = "boat1",
    boat2_name: str = "boat2",
    min_duration_sec: float = 30.0,
    sog_derivative_threshold: float = 0.2,
    smoothing_window: int = 300
) -> list[dict]:

    # Mono-bateau
    if boat2_changes is None or boat2_df is None:
        boat1_df['SOG_smoothed'] = boat1_df['SOG'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        boat1_df['SOG_derivative'] = boat1_df['SOG_smoothed'].diff(2).abs() / boat1_df['SecondsSince1970'].diff(2).abs()
        
        intervals = []
        changes = boat1_changes.sort_values('SecondsSince1970')
        for i in range(len(changes) - 1):
            start = changes.iloc[i]['SecondsSince1970']
            end = changes.iloc[i + 1]['SecondsSince1970']
            segment = boat1_df[(boat1_df['SecondsSince1970'] >= start) & (boat1_df['SecondsSince1970'] <= end)]
            if segment.empty:
                continue

            stable = segment[segment['SOG_derivative'] < sog_derivative_threshold]
            if stable.empty:
                continue

            duration = stable['SecondsSince1970'].max() - stable['SecondsSince1970'].min()
            if duration >= min_duration_sec:
                intervals.append({
                    'start_time': stable['SecondsSince1970'].min(),
                    'end_time': stable['SecondsSince1970'].max(),
                    'duration': duration,
                    'avg_SOG': stable['SOG'].mean(),
                    'SOG_variation': stable['SOG'].std(),
                    'stability_score': 1 / (1 + stable['SOG_derivative'].mean()),
                    'boat_name': boat1_name
                })

        return sorted(intervals, key=lambda x: (-x['duration'], -x['stability_score']))[:top_n]

    # Deux bateaux (version originale inchangée)
    boat1_changes['source'] = boat1_name
    boat2_changes['source'] = boat2_name
    
    # Combine and sort all change points chronologically
    all_changes = pd.concat([boat1_changes, boat2_changes], ignore_index=True)
    all_changes_sorted = all_changes.sort_values('SecondsSince1970').reset_index(drop=True)
    
    # Preprocess SOG data with smoothing and derivative calculation
    for df in [boat1_df, boat2_df]:
        # Smooth SOG data to reduce noise
        df['SOG_smoothed'] = df['SOG'].rolling(
            window=smoothing_window,
            min_periods=1,
            center=True
        ).mean()
        
        # Calculate robust derivative using central differences
        df['SOG_derivative'] = (
            df['SOG_smoothed'].diff(2) / 
            df['SecondsSince1970'].diff(2).abs()
        ).abs()
    intervals = []
    
    for i in range(len(all_changes_sorted) - 1):
        start = all_changes_sorted.loc[i, 'SecondsSince1970']
        end = all_changes_sorted.loc[i + 1, 'SecondsSince1970']
        
        # Get data for this interval
        boat1_interval = boat1_df[(boat1_df['SecondsSince1970'] >= start) & 
                                (boat1_df['SecondsSince1970'] <= end)].copy()
        boat2_interval = boat2_df[(boat2_df['SecondsSince1970'] >= start) & 
                                (boat2_df['SecondsSince1970'] <= end)].copy()
        
        if boat1_interval.empty or boat2_interval.empty:
            continue
            
        # Merge the two boats' data by time
        merged = pd.merge_asof(
            boat1_interval.sort_values('SecondsSince1970'),
            boat2_interval.sort_values('SecondsSince1970'),
            on='SecondsSince1970',
            suffixes=('_boat1', '_boat2')
        )
        
        # Find periods where both boats have stable speed
        stable_mask = (
            (merged['SOG_derivative_boat1'] < sog_derivative_threshold) &
            (merged['SOG_derivative_boat2'] < sog_derivative_threshold))
        
        # Group consecutive stable periods
        stable_groups = (stable_mask != stable_mask.shift(1)).cumsum()
        
        for group_id, group_data in merged[stable_mask].groupby(stable_groups):
            group_start = group_data['SecondsSince1970'].min()
            group_end = group_data['SecondsSince1970'].max()
            group_duration = group_end - group_start
            
            if group_duration >= min_duration_sec:
                # Calculate additional metrics
                avg_sog1 = group_data['SOG_boat1'].mean()
                avg_sog2 = group_data['SOG_boat2'].mean()
                
                intervals.append({
                    'start_time': group_start,
                    'end_time': group_end,
                    'duration': group_duration,
                    'boat1_name': boat1_name,
                    'boat2_name': boat2_name,
                    'avg_SOG_boat1': avg_sog1,
                    'avg_SOG_boat2': avg_sog2,
                    'SOG_variation_boat1': group_data['SOG_boat1'].std(),
                    'SOG_variation_boat2': group_data['SOG_boat2'].std(),
                    'stability_score': 1/(1 + group_data[['SOG_derivative_boat1', 
                                                       'SOG_derivative_boat2']].mean().mean())
                })
    
    # Sort by duration and stability score
    return sorted(intervals, 
                 key=lambda x: (-x['duration'], -x['stability_score']))[:top_n]


import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

"""def plot_sog_with_intervals(
    boat1_df: pd.DataFrame,
    boat2_df: pd.DataFrame,
    intervals: list[dict],
    boat1_name: str = "boat1",
    boat2_name: str = "boat2"
):
    plt.figure(figsize=(15, 8))
    
    time_offset = boat1_df['SecondsSince1970'].min()
    time1 = boat1_df['SecondsSince1970'] - time_offset
    time2 = boat2_df['SecondsSince1970'] - time_offset

    # --- Plot raw and smoothed SOG ---
    plt.plot(time1, boat1_df['SOG'], color='green', alpha=0.3, label=f'{boat1_name} Raw SOG')
    plt.plot(time1, boat1_df['SOG_smoothed'], color='green', linewidth=1.5, label=f'{boat1_name} Smoothed SOG')

    plt.plot(time2, boat2_df['SOG'], color='blue', alpha=0.3, label=f'{boat2_name} Raw SOG')
    plt.plot(time2, boat2_df['SOG_smoothed'], color='blue', linewidth=1.5, label=f'{boat2_name} Smoothed SOG')

    # --- Vertical lines for flat/stable intervals ---
    for i, interval in enumerate(intervals, 1):
        start = interval['start_time'] - time_offset
        end = interval['end_time'] - time_offset
        plt.axvline(start, color='red', linestyle='--', linewidth=1)
        plt.axvline(end, color='red', linestyle='--', linewidth=1)
        plt.text((start + end) / 2, plt.ylim()[1]*0.95, f"#{i}", color='red',
                 ha='center', va='top', fontsize=9, fontweight='bold', rotation=0)

    plt.xlabel('Time (seconds from start)')
    plt.ylabel('SOG (knots)')
    plt.title('SOG with Smoothed Curve and Stable Intervals')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()"""
"""
def plot_sog_with_intervals(boat1_df, boat2_df, intervals, boat1_name="boat1", boat2_name="boat2"):
    print(intervals)
    plt.figure(figsize=(15, 8))
    offset = boat1_df['SecondsSince1970'].min()
    time1 = boat1_df['SecondsSince1970'] - offset
    plt.plot(time1, boat1_df['SOG'], alpha=0.3, label=f'{boat1_name} Raw SOG')
    plt.plot(time1, boat1_df['SOG_smoothed'], color='green', linewidth=1.5, label=f'{boat1_name} Smoothed SOG')

    # Extraire les temps et valeurs pour min_SOG
    min_sog_times = [interval['maneuver_time'] - offset for interval in intervals]
    min_sog_values = [interval['min_SOG'] for interval in intervals]
    plt.plot(min_sog_times, min_sog_values, 'o--', color='orange', label=f'{boat1_name} Min SOG')
    # Tracer
    plt.plot(np.array(min_sog_times) - offset, min_sog_values, label=f'{boat1_name} Min SOG', linestyle='--', color='orange', marker='o')
"""

def plot_sog_with_intervals(boat1_df, boat2_df, intervals, boat1_name="boat1", boat2_name="boat2"):
    plt.figure(figsize=(15, 8))
    offset = boat1_df['SecondsSince1970'].min()
    time1 = boat1_df['SecondsSince1970'] - offset
    plt.plot(time1, boat1_df['SOG'], alpha=0.3, label=f'{boat1_name} Raw SOG')
    plt.plot(time1, boat1_df.get('SOG_smoothed', boat1_df['SOG']), label=f'{boat1_name} Smoothed SOG')

    if 'maneuver_time' in intervals[0]:
        min_sog_times = [interval['maneuver_time'] - offset for interval in intervals]
        min_sog_values = [interval['min_SOG'] for interval in intervals]
        plt.plot(min_sog_times, min_sog_values, 'o--', color='orange', label=f'{boat1_name} Min SOG')

    if boat2_df is not None:
        time2 = boat2_df['SecondsSince1970'] - offset
        plt.plot(time2, boat2_df['SOG'], alpha=0.3, label=f'{boat2_name} Raw SOG')
        plt.plot(time2, boat2_df.get('SOG_smoothed', boat2_df['SOG']), label=f'{boat2_name} Smoothed SOG')

    for i, interval in enumerate(intervals, 1):
        start = interval['start_time'] - offset
        end = interval['end_time'] - offset
        plt.axvline(start, color='red', linestyle='--')
        plt.axvline(end, color='red', linestyle='--')
        plt.text((start + end)/2, plt.ylim()[1]*0.95, f"#{i}", ha='center', color='red')

    plt.xlabel("Time (s)")
    plt.ylabel("SOG (knots)")
    plt.title("SOG and Stable Intervals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    if boat2_df is not None:
        time2 = boat2_df['SecondsSince1970'] - offset
        plt.plot(time2, boat2_df['SOG'], alpha=0.3, label=f'{boat2_name} Raw SOG')
        plt.plot(time2, boat2_df.get('SOG_smoothed', boat2_df['SOG']), label=f'{boat2_name} Smoothed SOG')
        for i, interval in enumerate(intervals, 1):
            start = interval['start_time'] - offset
            end = interval['end_time'] - offset
            plt.axvline(start, color='red', linestyle='--')
            plt.axvline(end, color='red', linestyle='--')
            plt.text((start + end)/2, plt.ylim()[1]*0.95, f"#{i}", ha='center', color='red')

        plt.xlabel("Time (s)")
        plt.ylabel("SOG (knots)")
        plt.title("SOG and Stable Intervals")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


"""def add_avg_twa_to_intervals(intervals: list[dict], boat1_df: pd.DataFrame, boat2_df: pd.DataFrame) -> list[dict]:
    for interval in intervals:
        start = interval['start_time']
        end = interval['end_time']

        # Moyenne TWA pour boat1
        mask1 = (boat1_df['SecondsSince1970'] >= start) & (boat1_df['SecondsSince1970'] <= end)
        twa_values1 = boat1_df.loc[mask1, 'TWA']
        interval['avg TWA boat1'] = twa_values1.mean() if not twa_values1.empty else None

        # Moyenne TWA pour boat2
        mask2 = (boat2_df['SecondsSince1970'] >= start) & (boat2_df['SecondsSince1970'] <= end)
        twa_values2 = boat2_df.loc[mask2, 'TWA']
        interval['avg TWA boat2'] = twa_values2.mean() if not twa_values2.empty else None

    return intervals"""

def add_avg_twa_to_intervals(intervals: list[dict], boat1_df: pd.DataFrame, boat2_df: pd.DataFrame = None) -> list[dict]:
    """
    Pour chaque intervalle, ajoute la moyenne de TWA pour chaque bateau séparément.
    """
    for interval in intervals:
        start = interval['start_time']
        end = interval['end_time']

        # Moyenne TWA pour boat1
        mask1 = (boat1_df['SecondsSince1970'] >= start) & (boat1_df['SecondsSince1970'] <= end)
        twa_values1 = boat1_df.loc[mask1, 'TWA']
        interval['avg TWA boat1'] = twa_values1.mean() if not twa_values1.empty else None

        # Moyenne TWA pour boat2 si disponible
        if boat2_df is not None:
            mask2 = (boat2_df['SecondsSince1970'] >= start) & (boat2_df['SecondsSince1970'] <= end)
            twa_values2 = boat2_df.loc[mask2, 'TWA']
            interval['avg TWA boat2'] = twa_values2.mean() if not twa_values2.empty else None

    return intervals


def load_boat_data(boat1_path: str, boat2_path: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """
    Load boat data from CSV files and extract boat names.
    Returns dataframes and corresponding boat names.
    """
    boat1_df = pd.read_csv(boat1_path)
    boat2_df = pd.read_csv(boat2_path)

    boat1_name = extract_boat_name(boat1_path)
    boat2_name = extract_boat_name(boat2_path)

    return boat1_df, boat2_df, boat1_name, boat2_name

"""def analyze_session(boat1_path: str, boat2_path: str) -> list[dict]:
    # Load data
    boat1_df, boat2_df, boat1_name, boat2_name = load_boat_data(boat1_path, boat2_path)

    # Detect COG change points
    boat1_changes = detect_COG_changes_rolling_mean(boat1_df)
    boat2_changes = detect_COG_changes_rolling_mean(boat2_df)

    # Plot full trajectories with COG changes
    plot_full_trajectories(boat1_df, boat2_df, boat1_changes, boat2_changes, boat1_name, boat2_name)

    # Compute longest intervals between change points
    longest_intervals = compute_longest_intervals(
        boat1_changes, boat2_changes,
        boat1_df, boat2_df,
        top_n=2,
        boat1_name=boat1_name,
        boat2_name=boat2_name
    )
    
    # Add average TWA per boat
    add_avg_twa_to_intervals(longest_intervals, boat1_df, boat2_df)

    # Plot the longest trajectory segments
    # plot_longest_segments(boat1_df, boat2_df, longest_intervals, boat1_name, boat2_name)
    plot_sog_with_intervals(boat1_df, boat2_df, longest_intervals, boat1_name, boat2_name)

    return longest_intervals"""

"""def analyze_session(boat1_path: str, boat2_path: str = None) -> list[dict]:
    if boat2_path is None:
        # Mono-bateau
        boat1_df = pd.read_csv(boat1_path)
        boat1_name = extract_boat_name(boat1_path)

        boat1_changes = detect_COG_changes_rolling_mean(boat1_df)
        plot_trajectories(boat1_df, boat1_changes, boat1_name=boat1_name)


        intervals = compute_longest_intervals(
            boat1_changes, None,
            boat1_df, None,
            top_n=2,
            boat1_name=boat1_name,
            boat2_name=None
        )

        add_avg_twa_to_intervals(intervals, boat1_df, None)
        plot_sog_with_intervals(boat1_df, None, intervals, boat1_name, None)

        return intervals

    else:
        # Deux bateaux
        boat1_df = pd.read_csv(boat1_path)
        boat2_df = pd.read_csv(boat2_path)
        boat1_name = extract_boat_name(boat1_path)
        boat2_name = extract_boat_name(boat2_path)

        boat1_changes = detect_COG_changes_rolling_mean(boat1_df)
        boat2_changes = detect_COG_changes_rolling_mean(boat2_df)

        plot_trajectories(boat1_df, boat2_df, boat1_changes, boat2_changes, boat1_name, boat2_name)

        intervals = compute_longest_intervals(
            boat1_changes, boat2_changes,
            boat1_df, boat2_df,
            top_n=2,
            boat1_name=boat1_name,
            boat2_name=boat2_name
        )

        add_avg_twa_to_intervals(intervals, boat1_df, boat2_df)
        plot_sog_with_intervals(boat1_df, boat2_df, intervals, boat1_name, boat2_name)

        return intervals
"""


def analyze_session(boat1_path: str, boat2_path: str = None) -> list[dict]:
    """
    Analyse une session avec un ou deux bateaux selon les fichiers fournis.
    Pour un seul bateau, détecte la première phase stable et analyse une fenêtre de 10 secondes autour.
    Pour deux bateaux, détecte les plus longues périodes de stabilité entre changements de cap.
    """
    boat1_df = pd.read_csv(boat1_path)
    boat1_name = extract_boat_name(boat1_path)

    if boat2_path is None:
        # Mono-bateau : nouvelle logique avec premier plateau
        boat1_changes = detect_COG_changes_rolling_mean(boat1_df)
        plot_trajectories(boat1_df, boat1_changes, boat1_name=boat1_name)

        intervals = detect_maneuvers_combined(boat1_df,boat1_changes)

        if intervals:
            for interval in intervals:
                interval['boat_name'] = boat1_name
            plot_sog_with_intervals(boat1_df, None, intervals, boat1_name, None)
            return intervals
        else:
            print("⚠️ Aucune manœuvre détectée.")
            return []


    else:
        # Deux bateaux : logique d'analyse complète
        boat2_df = pd.read_csv(boat2_path)
        boat2_name = extract_boat_name(boat2_path)

        boat1_changes = detect_COG_changes_rolling_mean(boat1_df)
        boat2_changes = detect_COG_changes_rolling_mean(boat2_df)

        plot_trajectories(boat1_df, boat1_changes,
                          boat2_df=boat2_df, boat2_changes=boat2_changes,
                          boat1_name=boat1_name, boat2_name=boat2_name)

        intervals = compute_longest_intervals(
            boat1_changes, boat2_changes,
            boat1_df, boat2_df,
            top_n=2,
            boat1_name=boat1_name,
            boat2_name=boat2_name
        )

        add_avg_twa_to_intervals(intervals, boat1_df, boat2_df)
        plot_sog_with_intervals(boat1_df, boat2_df, intervals, boat1_name, boat2_name)

        return intervals

"""
def find_first_plateau_window(
    boat_df: pd.DataFrame,
    sog_derivative_threshold: float = 0.1,
    smoothing_window: int = 50,
    margin_sec: float = 5.0
) -> dict | None:
    # Smooth and compute SOG derivative
    boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
    boat_df['SOG_derivative'] = boat_df['SOG_smoothed'].diff(2).abs() / boat_df['SecondsSince1970'].diff(2).abs()

    # Find first stable zone
    stable_mask = boat_df['SOG_derivative'] < sog_derivative_threshold
    groups = (stable_mask != stable_mask.shift()).cumsum()
    stable_groups = boat_df[stable_mask].groupby(groups)

    for _, group in stable_groups:
        if not group.empty:
            plateau_start = group['SecondsSince1970'].iloc[0]
            interval_start = plateau_start - margin_sec
            interval_end = plateau_start + margin_sec

            # Subset the data for the interval
            window_data = boat_df[
                (boat_df['SecondsSince1970'] >= interval_start) &
                (boat_df['SecondsSince1970'] <= interval_end)
            ]
            if window_data.empty:
                return None

            return {
                'start_time': interval_start,
                'end_time': interval_end,
                'duration': interval_end - interval_start,
                'avg_SOG': window_data['SOG'].mean(),
                'SOG_variation': window_data['SOG'].std(),
                'stability_score': 1 / (1 + window_data['SOG_derivative'].mean()),
            }

    return None  # No stable region found"""
"""
def find_first_plateau_window(
    boat_df: pd.DataFrame,
    sog_derivative_threshold: float = 0.3,
    smoothing_window: int = 70,
    margin_sec: float = 5.0,
    min_plateau_duration: float = 3.0  # en secondes
) -> dict | None:
    # Lissage et dérivée de la SOG
    boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
    boat_df['SOG_derivative'] = boat_df['SOG_smoothed'].diff(2).abs() / boat_df['SecondsSince1970'].diff(2).abs()

    # Recherche de zones stables
    stable_mask = boat_df['SOG_derivative'] < sog_derivative_threshold
    groups = (stable_mask != stable_mask.shift()).cumsum()
    stable_groups = boat_df[stable_mask].groupby(groups)

    for _, group in stable_groups:
        if not group.empty:
            duration = group['SecondsSince1970'].iloc[-1] - group['SecondsSince1970'].iloc[0]
            if duration >= min_plateau_duration:
                plateau_start = group['SecondsSince1970'].iloc[0]
                interval_start = plateau_start - margin_sec
                interval_end = plateau_start + margin_sec

                # Fenêtre autour du début du plateau
                window_data = boat_df[
                    (boat_df['SecondsSince1970'] >= interval_start) &
                    (boat_df['SecondsSince1970'] <= interval_end)
                ]
                if window_data.empty:
                    return None

                return {
                    'start_time': interval_start,
                    'end_time': interval_end,
                    'duration': interval_end - interval_start,
                    'avg_SOG': window_data['SOG'].mean(),
                    'SOG_variation': window_data['SOG'].std(),
                    'stability_score': 1 / (1 + window_data['SOG_derivative'].mean()),
                }

    return None  # Aucun plateau stable trouvé d'une durée suffisante
"""

from scipy.signal import argrelextrema
"""
def detect_maneuvers_from_sog_minima(boat_df: pd.DataFrame, order: int = 2, twa_threshold: float = 90.0) -> list[dict]:
    # Lisser la SOG si pas déjà fait
    if 'SOG_smoothed' not in boat_df.columns:
        boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=80, center=True, min_periods=1).mean()
    
    sog = boat_df['SOG_smoothed'].values

    # Trouver les indices des maxima et minima locaux
    maxima_idx = argrelextrema(sog, np.greater, order=order)[0]
    minima_idx = argrelextrema(sog, np.less, order=order)[0]

    maneuvers = []

    for min_idx in minima_idx:
        # Chercher le maximum précédent et suivant le minimum
        prev_max = [idx for idx in maxima_idx if idx < min_idx]
        next_max = [idx for idx in maxima_idx if idx > min_idx]

        if not prev_max or not next_max:
            continue  # ignorer si pas de max des deux côtés

        i1 = prev_max[-1]
        i2 = next_max[0]

        # Extraire l’intervalle de temps
        start_time = boat_df.iloc[i1]['SecondsSince1970']
        end_time = boat_df.iloc[i2]['SecondsSince1970']
        duration = end_time - start_time

        # Calculer variation de TWA
        twa_segment = boat_df.iloc[i1:i2+1]['TWA']
        delta_twa = abs(twa_segment.max()) - abs(twa_segment.min())

        maneuver_type = "virement" if delta_twa > twa_threshold else "empannage"

        maneuvers.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'min_SOG': sog[min_idx],
            'delta_TWA': delta_twa,
            'maneuver_type': maneuver_type,
        })

    return maneuvers"""


"""
def detect_maneuvers_from_sog_minima(
    boat_df: pd.DataFrame,
    order: int = 10,
    twa_threshold: float = 90.0
) -> list[dict]:

    if 'SOG_smoothed' not in boat_df.columns:
        boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=80, center=True, min_periods=1).mean()

    sog = boat_df['SOG_smoothed'].values
    maxima_idx = argrelextrema(sog, np.greater, order=order)[0]
    minima_idx = argrelextrema(sog, np.less, order=order)[0]

    maneuvers = []

    for min_idx in minima_idx:
        prev_max = [idx for idx in maxima_idx if idx < min_idx]
        next_max = [idx for idx in maxima_idx if idx > min_idx]

        if not prev_max or not next_max:
            continue

        i1 = prev_max[-1]
        i2 = next_max[0]

        start_time = boat_df.iloc[i1]['SecondsSince1970']
        end_time = boat_df.iloc[i2]['SecondsSince1970']
        duration = end_time - start_time

        twa_segment = boat_df.iloc[i1:i2+1]['TWA']
        delta_twa = abs(abs(twa_segment.max()) - abs(twa_segment.min()))

        # 🧭 Nouvelle logique de classification
        if delta_twa == 0:
            maneuver_type = "unknown"
        elif delta_twa < twa_threshold:
            maneuver_type = "virement"
        else:
            maneuver_type = "empannage"

        maneuvers.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'min_SOG': sog[min_idx],
            'delta_TWA': delta_twa,
            'maneuver_type': maneuver_type,
        })

    return maneuvers"""
"""
def detect_maneuvers_from_sog_minima(
    boat_df: pd.DataFrame,
    order: int = 2,
    twa_threshold: float = 90.0,
    min_duration: float = 1.0
) -> list[dict]:
    # Smooth SOG if not already done
    if 'SOG_smoothed' not in boat_df.columns:
        boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=80, center=True, min_periods=1).mean()

    sog = boat_df['SOG_smoothed'].values
    twa = boat_df['TWA'].values
    time = boat_df['SecondsSince1970'].values

    # Find extrema
    maxima_idx = argrelextrema(sog, np.greater, order=order)[0]
    minima_idx = argrelextrema(sog, np.less, order=order)[0]
    
    # Sort chronologically
    maxima_idx = sorted(maxima_idx)
    minima_idx = sorted(minima_idx)
    
    maneuvers = []
    
    # We'll process minima in order and find their surrounding maxima
    for i, min_idx in enumerate(minima_idx):
        # Find preceding maximum (must be before current min)
        prev_max = [m for m in maxima_idx if m < min_idx]
        if not prev_max:
            continue
        prev_max = prev_max[-1]  # take the last one before min_idx
        
        # Find following maximum (must be after current min)
        next_max = [m for m in maxima_idx if m > min_idx]
        if not next_max:
            continue
        next_max = next_max[0]  # take the first one after min_idx
        
        # Get time points and values
        start_time = time[prev_max]
        end_time = time[next_max]
        duration = end_time - start_time
        
        # Skip if duration is too short
        if duration < min_duration:
            continue
        
        # Get TWA values (using absolute values)
        twa_start = abs(twa[prev_max])
        twa_min = abs(twa[min_idx])
        twa_end = abs(twa[next_max])
        
        # Determine maneuver type
        if (twa_start < 15 and twa_end < 15):
            maneuver_type = "passage_bouée"
        elif (twa_start > twa_threshold and twa_end > twa_threshold):
            maneuver_type = "empannage"
        elif (twa_start < twa_threshold and twa_end < twa_threshold):
            maneuver_type = "virement"
        else:
            if abs(twa_end - twa_start) > twa_threshold:
                maneuver_type = "virement_empannage"
            else:
                maneuver_type = "changement_vitesse"
        
        maneuvers.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'min_SOG': sog[min_idx],
            'TWA_start': twa_start,
            'TWA_min': twa_min,
            'TWA_end': twa_end,
            'maneuver_type': maneuver_type,
        })

    # Now we need to handle cases where maxima are shared between maneuvers
    # by verifying that each maneuver's start_time >= previous maneuver's end_time
    if not maneuvers:
        return []
    
    # Sort by start time just in case
    maneuvers.sort(key=lambda x: x['start_time'])
    
    # Initialize the final list with the first maneuver
    final_maneuvers = [maneuvers[0]]
    
    for current in maneuvers[1:]:
        last = final_maneuvers[-1]
        
        # If current starts before last ends, we need to adjust
        if current['start_time'] < last['end_time']:
            # Choose which one to keep based on duration or other criteria
            if current['duration'] > last['duration']:
                # Replace the last one with current
                final_maneuvers[-1] = current
            # Else keep the last one and skip current
        else:
            final_maneuvers.append(current)
    
    return final_maneuvers"""

"""def detect_maneuvers_from_sog_minima(
    boat_df: pd.DataFrame,
    order: int = 5,
    twa_threshold: float = 90.0,
    min_duration: float = 5.0
) -> list[dict]:
    # Smooth SOG if not already done
    if 'SOG_smoothed' not in boat_df.columns:
        boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=100, center=True, min_periods=1).mean()

    sog = boat_df['SOG_smoothed'].values
    twa = boat_df['TWA'].values
    time = boat_df['SecondsSince1970'].values

    # Find extrema
    maxima_idx = argrelextrema(sog, np.greater, order=4)[0][1:] # Exclude the first point to avoid edge effects
    minima_idx = argrelextrema(sog, np.less, order=order)[0] #5 is good
    
    # Sort chronologically
    maxima_idx = sorted(maxima_idx)
    minima_idx = sorted(minima_idx)
    print(minima_idx)
    print("\n")
    print(maxima_idx)

    maneuvers = []
    
    # We'll process minima in order and find their surrounding maxima
    for i, min_idx in enumerate(minima_idx):
        # Find preceding maximum (must be before current min)
        prev_max = [m for m in maxima_idx if m < min_idx]
        if not prev_max:
            continue
        prev_max = prev_max[-1]  # take the last one before min_idx
        
        # Find following maximum (must be after current min)
        next_max = [m for m in maxima_idx if m > min_idx]
        if not next_max:
            continue
        next_max = next_max[0]  # take the first one after min_idx
        
        # Get time points and values
        start_time = time[prev_max]
        end_time = time[next_max]
        duration = end_time - start_time
        if duration < min_duration:
            continue
        # Get TWA values (using absolute values)
        twa_start = abs(twa[prev_max])
        twa_min = abs(twa[min_idx])
        twa_end = abs(twa[next_max])
        
        # Determine maneuver type
        if (twa_start < 15 and twa_end < 15):
            maneuver_type = "passage_bouée"
        elif (twa_start > twa_threshold and twa_end > twa_threshold):
            maneuver_type = "empannage"
        elif (twa_start < twa_threshold and twa_end < twa_threshold):
            maneuver_type = "virement"
        else:
            if abs(twa_end - twa_start) > twa_threshold:
                maneuver_type = "virement_empannage"
            else:
                maneuver_type = "changement_vitesse"
        
        maneuvers.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'min_SOG': sog[min_idx],
            'TWA_start': twa_start,
            'TWA_min': twa_min,
            'TWA_end': twa_end,
            'maneuver_type': maneuver_type,
            'maneuver_time': time[min_idx],
            'min_SOG': sog[min_idx],
        })

    return maneuvers
"""

def detect_maneuvers_combined(
    boat_df: pd.DataFrame,
    cog_changes: pd.DataFrame,
    sog_order: int = 5,
    twa_threshold: float = 90.0,
    min_duration: float = 5.0,
    time_window_sec: float = 10.0
) -> list[dict]:
    if 'SOG_smoothed' not in boat_df.columns:
        boat_df['SOG_smoothed'] = boat_df['SOG'].rolling(window=120, center=True, min_periods=1).mean()

    sog = boat_df['SOG_smoothed'].values
    twa = boat_df['TWA'].values
    time = boat_df['SecondsSince1970'].values

    maxima_idx = argrelextrema(sog, np.greater, order=4)[0][1:]
    minima_idx = argrelextrema(sog, np.less, order=sog_order)[0]
    cog_change_times = cog_changes['SecondsSince1970'].values

    maneuvers = []
    
    for min_idx in minima_idx:
        min_time = time[min_idx]
        
        # Check for COG change nearby
        if not np.any(np.abs(cog_change_times - min_time) <= time_window_sec):
            continue  # Skip if no COG change near this SOG min

        prev_max = [m for m in maxima_idx if m < min_idx]
        next_max = [m for m in maxima_idx if m > min_idx]
        if not prev_max or not next_max:
            continue

        i1, i2 = prev_max[-1], next_max[0]
        start_time, end_time = time[i1], time[i2]
        duration = end_time - start_time
        if duration < min_duration:
            continue

        twa_start = abs(twa[i1])
        twa_min = abs(twa[min_idx])
        twa_end = abs(twa[i2])

        # Maneuver classification
        if twa_start < 15 and twa_end < 15:
            maneuver_type = "passage_bouée"
        elif twa_start > twa_threshold and twa_end > twa_threshold:
            maneuver_type = "empannage"
        elif twa_start < twa_threshold and twa_end < twa_threshold:
            maneuver_type = "virement"
        else:
            maneuver_type = "virement_empannage" if abs(twa_end - twa_start) > twa_threshold else "changement_vitesse"

        maneuvers.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'TWA_start': twa_start,
            'TWA_min': twa_min,
            'TWA_end': twa_end,
            'maneuver_type': maneuver_type,
            'maneuver_time': min_time,
            'min_SOG': sog[min_idx]
        })

    return maneuvers
