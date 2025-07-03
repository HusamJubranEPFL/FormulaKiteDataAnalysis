import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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


def plot_full_trajectories(boat1_df, boat2_df, boat1_changes, boat2_changes, boat1_name="boat1", boat2_name="boat2"):

    """
    Plot full trajectories of both boat1 and boat2 with markers on COG change points.
    """
    colors = {boat1_name: 'green', boat2_name: 'blue'}

    plt.figure(figsize=(10, 8))
    plt.scatter(boat2_df['Lon'], boat2_df['Lat'], c=colors[boat2_name], marker='x', s=10, label=f'Trajectory {boat2_name}')
    plt.scatter(boat1_df['Lon'], boat1_df['Lat'], c=colors[boat1_name], marker='x', s=10, label=f'Trajectory {boat1_name}')
    plt.scatter(boat2_changes['Lon'], boat2_changes['Lat'], c='red', s=40, label=f'COG Change Points {boat2_name}')
    plt.scatter(boat1_changes['Lon'], boat1_changes['Lat'], c='red', s=40, label=f'COG Change Points {boat1_name}')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectories with COG Change Points')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compute_longest_intervals(boat1_changes: pd.DataFrame, boat2_changes: pd.DataFrame, top_n: int = 2,boat1_name="boat1", boat2_name="boat2") -> list[dict]:
    """
    Combine COG change points and compute the top N longest time intervals between them.
    """
    boat1_changes['source'] = 'boat1'
    boat2_changes['source'] = 'boat2'
    all_changes = pd.concat([boat1_changes, boat2_changes], ignore_index=True)
    all_changes_sorted = all_changes.sort_values('SecondsSince1970').reset_index(drop=True)


    intervals = []
    for i in range(len(all_changes_sorted) - 1):
        start = all_changes_sorted.loc[i, 'SecondsSince1970'] +5 # Adding a 5 seconds buffer to start time
        end = all_changes_sorted.loc[i + 1, 'SecondsSince1970'] -5 # Adding a 5 seconds buffer to end time
        intervals.append({
            'duration': end - start,
            'start_time': start,
            'end_time': end,
            'boat1_name': boat1_name,
            'boat2_name': boat2_name
        })
    return sorted(intervals, key=lambda x: x['duration'], reverse=True)[:top_n]

def add_avg_twa_to_intervals(intervals: list[dict], boat1_df: pd.DataFrame, boat2_df: pd.DataFrame) -> list[dict]:
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

        # Moyenne TWA pour boat2
        mask2 = (boat2_df['SecondsSince1970'] >= start) & (boat2_df['SecondsSince1970'] <= end)
        twa_values2 = boat2_df.loc[mask2, 'TWA']
        interval['avg TWA boat2'] = twa_values2.mean() if not twa_values2.empty else None

    return intervals

def plot_longest_segments(boat1_df, boat2_df, intervals, boat1_name="boat1", boat2_name="boat2"):
    """
    Plot the two longest trajectory segments between COG change points,
    using different linewidths and showing avg TWA per boat in the legend.
    """
    colors = {boat1_name: 'green', boat2_name: 'blue'}
    linewidths = [1, 3]  # Different thicknesses per interval

    plt.figure(figsize=(10, 8))

    for idx, interval in enumerate(intervals, 1):
        t_min, t_max = interval['start_time'], interval['end_time']
        twa1 = interval.get('avg TWA boat1')
        twa2 = interval.get('avg TWA boat2')
        twa1_str = f"{twa1:.1f}°" if twa1 is not None else "N/A"
        twa2_str = f"{twa2:.1f}°" if twa2 is not None else "N/A"

        traj_boat1 = boat1_df[(boat1_df['SecondsSince1970'] >= t_min) & (boat1_df['SecondsSince1970'] <= t_max)]
        traj_boat2 = boat2_df[(boat2_df['SecondsSince1970'] >= t_min) & (boat2_df['SecondsSince1970'] <= t_max)]

        lw = linewidths[(idx - 1) % len(linewidths)]

        plt.plot(traj_boat2['Lon'], traj_boat2['Lat'], color=colors[boat2_name], linewidth=lw,
                 label=f'{boat2_name} (interval {idx}, avg TWA: {twa2_str})')
        plt.plot(traj_boat1['Lon'], traj_boat1['Lat'], color=colors[boat1_name], linewidth=lw,
                 label=f'{boat1_name} (interval {idx}, avg TWA: {twa1_str})')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Top 2 Longest Segments Between COG Changes")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def analyze_session(boat1_path: str, boat2_path: str) -> list[dict]:
    """
    Perform a full analysis of a sailing session given CSV paths for boat1 and boat2.
    Returns a list of the top N longest intervals with avg TWA per boat.
    """
    # Load data
    boat1_df = pd.read_csv(boat1_path)
    boat2_df = pd.read_csv(boat2_path)

    # Extract boat names from file names
    boat1_name = extract_boat_name(boat1_path)
    boat2_name = extract_boat_name(boat2_path)

    # Detect COG change points
    boat1_changes = detect_COG_changes_rolling_mean(boat1_df)
    boat2_changes = detect_COG_changes_rolling_mean(boat2_df)

    # Plot full trajectories with COG changes
    plot_full_trajectories(boat1_df, boat2_df, boat1_changes, boat2_changes, boat1_name, boat2_name)

    # Compute longest intervals between change points
    longest_intervals = compute_longest_intervals(
        boat1_changes, boat2_changes,
        top_n=2,
        boat1_name=boat1_name,
        boat2_name=boat2_name
    )

    # Add average TWA per boat
    add_avg_twa_to_intervals(longest_intervals, boat1_df, boat2_df)

    # Print interval summary
    for idx, interval in enumerate(longest_intervals, 1):
        twa1 = interval.get('avg TWA boat1')
        twa2 = interval.get('avg TWA boat2')
        twa1_str = f"{twa1:.1f}°" if twa1 is not None else "N/A"
        twa2_str = f"{twa2:.1f}°" if twa2 is not None else "N/A"

        print(f"Interval {idx}:")
        print(f"  Start time: {interval['start_time']}")
        print(f"  End time  : {interval['end_time']}")
        print(f"  Duration  : {interval['duration']} seconds")
        print(f"  Avg TWA {boat1_name}: {twa1_str}")
        print(f"  Avg TWA {boat2_name}: {twa2_str}")
        print()

    # Plot the longest trajectory segments
    plot_longest_segments(boat1_df, boat2_df, longest_intervals, boat1_name, boat2_name)

    return longest_intervals
