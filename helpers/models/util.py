"""
util.py

Provides utilization metric calculations

Last Updated: 2025_04_06
"""
from typing import List, Tuple
import pandas as pd
from datetime import datetime

def build_processed_df(df: pd.DataFrame, indices: list, start_times: list[datetime], end_times: list[datetime],
                       server_ids: list[int]) -> pd.DataFrame:
    """
    Create a processed df with server metrics

    :param df: Subset of original DataFrame
    :type df: pd.DataFrame
    :param indices: Selected rows
    :type indices: list
    :param start_times: Start times per transaction
    :type start_times: list[datetime]
    :param end_times: End times per transaction
    :type end_times: list[datetime]
    :param server_ids: Server ids
    :type server_ids: list[int]
    :return: DataFrame with computed columns
    :rtype: pd.DataFrame
    """
    # All not blocked records
    processed_df = df.loc[indices].copy()

    # Building df
    processed_df['StartTime'] = start_times
    processed_df['EndTime'] = end_times
    processed_df['WaitTime'] = (processed_df['StartTime'] - processed_df['ArrivalTime']).dt.total_seconds() / 60
    processed_df['SystemTime'] = (processed_df['EndTime'] - processed_df['ArrivalTime']).dt.total_seconds() / 60
    processed_df['ServerID'] = server_ids
    return processed_df


def calculate_utilization(server_logs: List[List[Tuple[datetime, datetime]]], simulation_start: datetime,
                          simulation_end: datetime) -> pd.DataFrame:
    util_data = []

    # Total running time in minutes
    total_sim_time = (simulation_end - simulation_start).total_seconds() / 60

    for sid, log in enumerate(server_logs):
        # Sum busy times for specific server
        busy_time = sum((et - st).total_seconds() for st, et in log) / 60

        # Remaining time (not working/busy)
        idle_time = total_sim_time - busy_time

        # Busy over overall time
        utilization = busy_time / total_sim_time if total_sim_time > 0 else 0.0

        # Append specific server metrics
        util_data.append({
            'ServerID': sid,
            'BusyTime(min)': round(busy_time, 2),
            'IdleTime(min)': round(idle_time, 2),
            'Utilization(%)': round(utilization * 100, 2)
        })

    return pd.DataFrame(util_data)
