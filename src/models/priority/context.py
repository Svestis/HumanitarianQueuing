"""
context.py
context-aware queuing

Prioritizing individuals based on a dynamic context score that considers vulnerability, wait time, age,
and service efficiency.

Last Updated: 2025-04-06
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from pandas import Timestamp

from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def compute_context_score(row: pd.Series, current_time: pd.Timestamp) -> float:
    """
    Compute a context score based on vulnerability, age, wait time, and transaction efficiency.
    score = 0.4 * Vulnerability + 0.2 * NormalizedWaitTime + 0.2 * Age + 0.2 * (1 / TransactionTime)
    :param row: A row from the dataset
    :type row: pd.Series
    :param current_time: The current simulation time
    :type current_time: pd.Timestamp
    :return: Score representing priority
    :rtype: float
    """
    wait_time: float = max((current_time - row['ArrivalTime']).total_seconds() / 60, 0)

    # Normalize inputs
    vulnerability: float = row['VulnerabilityScore']
    age: int = row['Age'] / 100  # Max age = 100
    trans_time: float = 1 / max(row['TransactionTime'], 1)  # Inverse of transaction time
    wait_norm: float = wait_time / 60  # Assume 1 hour normalization

    return 0.4 * vulnerability + 0.2 * wait_norm + 0.2 * age + 0.2 * trans_time


def cas(num_servers: int, original_df: pd.DataFrame = None, capacity: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Context-Aware Scheduling

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

    # Sort arrival times
    df: pd.DataFrame = df.sort_values('ArrivalTime')

    # Starts working on first arrival
    current_time = df['ArrivalTime'].min()

    # Init server times
    server_end_times: list[int] = [current_time] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]

    unarrived: list = list(df.iterrows())
    queue, processed_indices, start_times, end_times, server_assigned = [], [], [], [], []
    in_system: list = []
    blocked_count: int = 0

    while unarrived or queue:

        # Newly arrived individuals
        while unarrived and unarrived[0][1]['ArrivalTime'] <= current_time:
            queue.append(unarrived.pop(0))

        # Completed service
        in_system = [et for et in in_system if et > current_time]

        if capacity is not None:
            while len(queue) + len(in_system) > capacity:
                queue.pop(0)
                blocked_count += 1

        # Any server free assigned
        if queue and any(end <= current_time for end in server_end_times):
            # Score all queue clients
            scored = sorted(
                [(i, compute_context_score(row, current_time), row['ArrivalTime']) for i, (_, row) in enumerate(queue)],
                key=lambda x: (-x[1], x[2])
            )

            # Best to assign
            best_i: int = scored[0][0]

            # Asisgn next to queue
            original_idx, row = queue.pop(best_i)

            # Assign to earliest available server
            server_idx: np.signedinteger = np.argmin(server_end_times)

            # Actual start time
            start_time: pd.Timestamp = max(current_time, row['ArrivalTime'], server_end_times[server_idx])

            # Actual end time
            end_time: Timestamp = start_time + timedelta(minutes=row['TransactionTime'])

            # Update server availability
            server_end_times[server_idx]: list[datetime] = end_time

            # logging
            server_logs[server_idx].append((start_time, end_time))

            in_system.append(end_time); start_times.append(start_time); end_times.append(end_time)
            server_assigned.append(server_idx); processed_indices.append(original_idx)

        # Advance time
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

    # updated df
    processed_df = build_processed_df(df, processed_indices, start_times, end_times, server_assigned)

    # update metrics
    simulation_start: datetime = processed_df['ArrivalTime'].min()
    simulation_end: datetime = processed_df['EndTime'].max()

    # Server util df
    util_df: pd.DataFrame = calculate_utilization(server_logs, simulation_start, simulation_end)

    # Metrics
    metrics: dict = calculate_metrics(processed_df, server_logs, blocked_count, total_arrivals=total_arrivals,
                               simulation_start=simulation_start, simulation_end=simulation_end)

    return processed_df, util_df, server_logs, metrics
