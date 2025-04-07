"""
ros.py

Implementing queue where individuals are served in random order when servers become available.

Last Updated: 2025-04-06
"""
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Tuple, Optional
from pandas import Timestamp
from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def ros(num_servers: int, original_df: pd.DataFrame = None, capacity: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Simulates a random order service queue.

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
    random.seed(42)

    if original_df.empty:
        return pd.DataFrame(), pd.DataFrame(), [], default_metric_output()

    df: pd.DataFrame = original_df.copy()
    total_arrivals: int = len(original_df)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format="%H:%M")

    # Sort arrival times descending
    df = df.sort_values(by='ArrivalTime').reset_index(drop=True)

    # Init server times
    server_end_times: list[pd.Timestamp] = [df['ArrivalTime'].min()] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]
    start_times, end_times, server_assigned = [], [], []
    blocked_count: int = 0
    in_system: list = []
    queue: list = []
    current_idx: int = 0
    current_time: int = df['ArrivalTime'].min()

    while current_idx < len(df) or queue:

        # Add new arrivals to queue
        while current_idx < len(df) and df.loc[current_idx, 'ArrivalTime'] <= current_time:
            if capacity is not None and len(in_system) >= capacity:
                blocked_count += 1
            else:
                queue.append(df.loc[current_idx])
            current_idx += 1

        # Assign servers to randomly selected clients
        for sid in range(num_servers):

            # Free server + queue has customers
            if queue and server_end_times[sid] <= current_time:

                # Random selection
                random.shuffle(queue)
                row: pd.Series = queue.pop(0)

                # Actual start time (filtering for service before arrival)
                arrival_time = row['ArrivalTime']
                start_time: pd.Timestamp = max(arrival_time, server_end_times[sid])

                # Actual end time
                end_time: Timestamp = start_time + timedelta(minutes=int(row['TransactionTime']))

                # Update server availability
                server_end_times[sid]: list[datetime] = end_time

                # logging
                server_logs[sid].append((start_time, end_time))

                in_system.append(end_time); start_times.append(start_time); end_times.append(end_time)
                server_assigned.append(sid)

        current_time += timedelta(minutes=1)

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
