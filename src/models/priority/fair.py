"""
fair.py
Reduces bias by giving higher service priority to individuals with higher vulnerability scores.

Last Updated: 2025-04-06
"""
import datetime
from datetime import timedelta
from typing import Optional, Tuple
import pandas as pd
from src import build_processed_df, calculate_utilization, calculate_metrics, default_metric_output

def fair(num_servers: int, original_df: pd.DataFrame = None, capacity: Optional[int] = None)\
        -> Tuple[pd.DataFrame, pd.DataFrame, list[list[tuple[datetime, datetime]]], dict]:
    """
    Fair queuing based on inverse vulnerability score.

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

    # Sort arrival times
    df = df.sort_values(by='ArrivalTime')

    # Init server times
    server_end_times: list[pd.Timestamp] = [df['ArrivalTime'].min()] * num_servers

    # inits
    server_logs: list[list[tuple[datetime, datetime]]] = [[] for _ in range(num_servers)]
    start_times, end_times, server_assigned = [], [], []
    blocked_count: int = 0
    in_system: list = []

    queue: list = []
    time: pd.Timestamp = df['ArrivalTime'].min()
    records: list[dict] = df.to_dict('records')
    i: int = 0

    # Iter through the data
    while i < len(records) or queue:

        # Add new arrivals to queue
        while i < len(records) and records[i]['ArrivalTime'] <= time:
            record: dict = records[i]
            weight: float = 1 - record['VulnerabilityScore']  # Lower = higher priority
            queue.append((weight, i, record))
            i += 1

        # Sort by priority
        queue.sort(key=lambda x: x[0])  # lower = higher priority

        for sid in range(num_servers):
            if not queue:
                break

            if server_end_times[sid] <= time:
                _, rec_idx, record = queue.pop(0)


                in_system: list = [et for et in in_system if et > time] # People being served or waiting

                # Capacity
                if capacity is not None and len(in_system) >= capacity:
                    blocked_count += 1
                    continue

                # Actual start time (filtering for service before arrival)
                arrival_time = record['ArrivalTime']
                start_time = max(arrival_time, server_end_times[sid])

                # Actual end time
                end_time = start_time + timedelta(minutes=record['TransactionTime'])

                # Update server availability
                server_end_times[sid]: list[datetime] = end_time

                # logging
                server_logs[sid].append((start_time, end_time))

                in_system.append(end_time); start_times.append(start_time); end_times.append(end_time)
                server_assigned.append(sid)

        time += timedelta(minutes=1)

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