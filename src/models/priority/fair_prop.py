"""
fair_prop.py
Proportional Fairness

Based on a fairness score: vulnerability / transaction time.

Last Updated: 2025-04-06
"""
from typing import Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def proportional_fairness(num_servers: int, original_df: pd.DataFrame = None, capacity: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Implements a proportional fairness queuing
    Fairness is determined by vulnerability score and transaction time.

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
    total_arrivals = len(original_df)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format="%H:%M")

    # Sort arrivals
    df = df.sort_values(by='ArrivalTime')

    # Init server times
    server_end_times: list[pd.Timestamp] = [df['ArrivalTime'].min()] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]
    start_times, end_times, server_assigned = [], [], []
    blocked_count: int = 0
    in_system: list = []
    current_time = df['ArrivalTime'].min()
    unarrived = list(df.iterrows())
    processed_indices, queue = [], []

    while unarrived or queue:

        # Move arrivals into the queue
        while unarrived and unarrived[0][1]['ArrivalTime'] <= current_time:
            idx, row = unarrived.pop(0)
            score = row['VulnerabilityScore'] / max(row['TransactionTime'], 1e-3)
            queue.append((score, idx))

        # Remove individuals who have finished service
        in_system = [et for et in in_system if et > current_time]

        # Enforce capacity limit
        if capacity is not None:
            while len(queue) + len(in_system) > capacity:
                queue.pop()  # remove lowest score
                blocked_count += 1

        # If server is free and queue not empty
        if queue and any(end <= current_time for end in server_end_times):
            queue.sort(key=lambda x: (-x[0], x[1]))  # sort descending by score
            score, idx = queue.pop(0)
            row = df.loc[idx]

            server_idx = np.argmin(server_end_times)
            start_time = max(current_time, row['ArrivalTime'], server_end_times[server_idx])
            end_time = start_time + timedelta(minutes=int(row['TransactionTime']))

            server_end_times[server_idx] = end_time
            server_logs[server_idx].append((start_time, end_time))
            in_system.append(end_time)

            start_times.append(start_time)
            end_times.append(end_time)
            server_assigned.append(server_idx)
            processed_indices.append(idx)

        # Move to next event
        next_times = []
        if unarrived:
            next_times.append(unarrived[0][1]['ArrivalTime'])
        if queue:
            next_times.extend(server_end_times)
        if next_times:
            current_time = min(next_times)
        else:
            break

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
