"""
priority.py

Priority-based queuing where higher vulnerability scores are served first, with tie-branking

Last Updated: 2025_04_06
"""
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from pandas import Timestamp

from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def priority(num_servers: int, original_df: pd.DataFrame = None, capacity: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Simulate a priority queue based on vulnerability with tie-braking

    :param num_servers: Number of parallel servers
    :type num_servers: int
    :param original_df: The original df, passed by another function
    :type original_df: pd.Dataframe
    :param capacity: The system's capacity before blocking
    :type capacity: Optional[int]
    :return: A tuple containing a dataFrame with customer detais, a dataFrame with server utilization statistics,
    a dictionary with performance metrics
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]
    """
    if original_df.empty:
        return pd.DataFrame(), pd.DataFrame(), [], default_metric_output()

    df: pd.DataFrame = original_df.copy()
    total_arrivals: int = len(original_df)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format="%H:%M")

    # Sort by arrival time first, then by descending vulnerability score (higher score = higher priority)
    df.sort_values(by=['ArrivalTime', 'VulnerabilityScore'], ascending=[True, False], inplace=True)

    # Init server times
    server_end_times: list[pd.Timestamp] = [df['ArrivalTime'].min()] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]
    start_times, end_times, server_assigned = [], [], []
    blocked_count: int = 0
    in_system: list = []


    for idx, row in df.iterrows():
        current_time: Timestamp = row['ArrivalTime']
        in_system = [et for et in in_system if et > current_time]

        if capacity is not None and len(in_system) >= capacity:
            blocked_count += 1
            continue

        server_idx: np.signedinteger = np.argmin(server_end_times)

        # Start time = max of arrival or when the server is free
        arrival_time: Timestamp = row['ArrivalTime']
        start_time: Timestamp = max(arrival_time, server_end_times[server_idx])
        end_time: Timestamp = start_time + timedelta(minutes=row['TransactionTime'])

        # Update state
        server_end_times[server_idx] = end_time
        server_logs[server_idx].append((start_time, end_time))

        in_system.append(end_time)
        start_times.append(start_time)
        end_times.append(end_time)
        server_assigned.append(server_idx)

    if not start_times:
        return pd.DataFrame(), pd.DataFrame(), [], default_metric_output()

    # update df
    indices = df.index[:len(start_times)]
    processed_df = build_processed_df(df.loc[indices], indices, start_times, end_times, server_assigned)

    # update metrics
    simulation_start: datetime = processed_df['ArrivalTime'].min()
    simulation_end: datetime = processed_df['EndTime'].max()

    # Server Util df
    util_df: pd.DataFrame = calculate_utilization(server_logs, simulation_start, simulation_end)

    # Metrics
    metrics: dict = calculate_metrics(processed_df, server_logs, blocked_count, total_arrivals=total_arrivals,
                               simulation_start=simulation_start, simulation_end=simulation_end)

    return processed_df, util_df, server_logs, metrics
