"""
lifo.py

Implementing LIFO queue and calculate relevant metrics

Last Updated: 2025-04-06
"""
from datetime import datetime, timedelta
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from pandas import Timestamp
from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def lifo(num_servers: int, original_df: pd.DataFrame = None, capacity: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Simulate a LIFO multi-server queue M/M/c/K

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
    if original_df.empty :
        return pd.DataFrame(), pd.DataFrame(), [], default_metric_output()

    df: pd.DataFrame = original_df.copy()
    total_arrivals: int = len(original_df)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format="%H:%M")

    # Sort arrival times descending
    df: pd.DataFrame = df.sort_values(by='ArrivalTime', ascending=False).copy()

    # Init server times
    server_end_times: list[pd.Timestamp] = [df['ArrivalTime'].min()] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]
    start_times, end_times, server_assigned = [], [], []
    blocked_count: int = 0
    in_system: list = []

    # iter through the data
    for idx, row in df.iterrows():
        current_time: datetime = row['ArrivalTime']

        in_system: list = [et for et in in_system if et > current_time] # People being served or waiting

        if capacity is not None and len(in_system) >= capacity:
            blocked_count += 1
            continue

        # Server availability
        server_idx: np.signedinteger = np.argmin(server_end_times)

        # Actual start time (filtering for service before arrival)
        arrival_time: pd.Timestamp = row['ArrivalTime']
        start_time: pd.Timestamp = max(arrival_time, server_end_times[server_idx])

        # Actual end time
        end_time: Timestamp = start_time + timedelta(minutes=row['TransactionTime'])

        # Update server availability
        server_end_times[server_idx]: list[datetime] = end_time

        # logging
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

    # Server Util df
    util_df: pd.DataFrame = calculate_utilization(server_logs, simulation_start, simulation_end)

    metrics: dict = calculate_metrics(processed_df, server_logs, blocked_count, total_arrivals=total_arrivals,
                               simulation_start=simulation_start, simulation_end=simulation_end)

    return processed_df, util_df, server_logs, metrics