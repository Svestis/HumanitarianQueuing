"""
community.py
Community-Aware Queuing

Predefined community group priorities.

Last Updated: 2025-04-01
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pandas import Timestamp
from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def community_aware(original_df: pd.DataFrame, num_servers: int, community_priority: dict[str, int],
                    capacity: int = None)\
        -> tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Community-aware queuing strategy. Higher priority communities are served first.

    :param num_servers: Number of parallel servers
    :type num_servers: int
    :param original_df: The original df, passed by another function
    :type original_df: pd.Dataframe
    :param community_priority: Mapping communities to priority levels (lower value = higher priority).
    :type community_priority: dict[str, int]
    :param capacity: The system's capacity before blocking
    :type capacity: Optional[int]
    :return: A tuple containing a dataFrame with customer detais, a dataFrame with server utilization statistics,
    a dictionary with performance metrics
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]
    """
    if original_df.empty:
        return pd.DataFrame(), pd.DataFrame(), [], default_metric_output()

    df: pd.DataFrame = original_df.copy()
    total_arrivals: int = len(df)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format="%H:%M")

    # Assign random communities (if not already present)
    if 'Community' not in df.columns:
        df['Community'] = np.random.choice(['CommunityA', 'CommunityB'], size=len(df))

    # Assigning prriority values
    df['PriorityValue'] = df['Community'].map(community_priority)
    df = df.sort_values(by=['PriorityValue', 'ArrivalTime'])

    # Init srver times
    server_end_times: list[pd.Timestamp] = [df['ArrivalTime'].min()] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]
    start_times, end_times, server_assigned = [], [], []
    blocked_count: int = 0
    in_system: list = []

    # iter through the data
    for idx, row in df.iterrows():

        current_time: datetime = row['ArrivalTime']

        in_system: list = [et for et in in_system if et > current_time]

        if capacity is not None and len(in_system) >= capacity:
            blocked_count += 1
            continue

        # Server availability
        server_idx: np.signedinteger = np.argmin(server_end_times)

        # Actual start time
        arrival_time: pd.Timestamp = row['ArrivalTime']
        start_time: pd.Timestamp = max(arrival_time, server_end_times[server_idx])

        # Actual end time
        end_time: Timestamp = start_time + timedelta(minutes=row['TransactionTime'])

        # Update server availability
        server_end_times[server_idx]: list[datetime] = end_time

        # Logging
        server_logs[server_idx].append((start_time, end_time))

        in_system.append(end_time); start_times.append(start_time); end_times.append(end_time)
        server_assigned.append(server_idx)

    if not start_times:
        return pd.DataFrame(), pd.DataFrame(), [], default_metric_output()

    # update df
    indices = df.index[:len(start_times)]
    processed_df = build_processed_df(df.loc[indices], indices, start_times, end_times, server_assigned)

    # update metrics
    simulation_start: datetime = processed_df['ArrivalTime'].min()
    simulation_end: datetime = processed_df['EndTime'].max()

    # Server util df
    util_df: pd.DataFrame = calculate_utilization(server_logs, simulation_start, simulation_end)

    metrics: dict = calculate_metrics(processed_df, server_logs, blocked_count=blocked_count,
                                      total_arrivals=total_arrivals, simulation_start=simulation_start,
                                      simulation_end=simulation_end)

    return processed_df, util_df, server_logs, metrics
