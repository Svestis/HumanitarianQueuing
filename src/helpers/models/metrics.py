"""
metrics.py

Provides queueing performance metric calculations

Last Updated: 2025_04_06
"""
import pandas as pd
from datetime import datetime

def calculate_metrics(df: pd.DataFrame, server_logs: list, blocked_count: int,
                      total_arrivals: int, simulation_start: datetime,
                      simulation_end: datetime) -> dict:
    """
    Calculate queueing performance metrics
    :param df: DataFrame containing processed records
    :type df: pd.DataFrame
    :param server_logs: Start times and end times per server
    :type server_logs: list[list[tuple[datetime, datetime]]]
    :param blocked_count: Number of individuals who could not be served due to capacity limits
    :type blocked_count: int
    :param total_arrivals: Total number of arrivals (served and blocked)
    :type total_arrivals: int
    :param simulation_start: The earliest arrival time
    :type simulation_start: datetime
    :param simulation_end: The latest end time
    :type simulation_end: datetime
    :return: Metrics
    :rtype: dict
    """
    # Total time in minutes
    total_sim_time: float = (simulation_end - simulation_start).total_seconds() / 60

    # Assuring no zero division
    total_sim_time = float(max(float(total_sim_time), 1e-6))

    # Total idle time across all servers
    idle_time = df['IdleTime'].sum() if 'IdleTime' in df.columns else 0

    # Minutes
    avg_wait: float = df['WaitTime'].mean()

    # Waiting + service time in minutes
    avg_system_time: float = df['SystemTime'].mean()

    # Served per minutes
    throughput: float = len(df) / total_sim_time

    # Little's law for average number of people in queue
    avg_queue_length: float = avg_wait * throughput

    # Little's law for average people in the system
    avg_number_in_system: float = avg_system_time * throughput

    # Probability of waiting
    p_wait: float = (df['WaitTime'] > 0).mean()

    # Propability of blocking for exceeding capacity
    p_block: float = blocked_count / total_arrivals if total_arrivals > 0 else 0.0

    # Total time working for all servers
    busy_time = df['TransactionTime'].sum()

    return {
        'AverageWaitingTime(W)': round(avg_wait, 2),
        'AverageSystemTime(Ws)': round(avg_system_time, 2),
        'Throughput': round(throughput, 2),
        'AverageQueueLength(Lq)': round(avg_queue_length, 2),
        'AverageNumberInSystem(Ls)': round(avg_number_in_system, 2),
        'AverageUtilization(%)': round(100 * busy_time / (total_sim_time * len(server_logs)), 2),
        'TotalIdleTime(min)': round(idle_time, 2),
        'TotalBusyTime(min)': round(busy_time, 2),
        'ProbabilityOfWaiting(Pwait)': round(p_wait, 2),
        'ProbabilityOfBlocking(Pblock)': round(p_block, 2)
    }

def default_metric_output() -> dict:
    return {
        'AverageWaitingTime(W)': 0.0,
        'AverageSystemTime(Ws)': 0.0,
        'Throughput': 0.0,
        'AverageQueueLength(Lq)': 0.0,
        'AverageNumberInSystem(Ls)': 0.0,
        'AverageUtilization(%)': 0.0,
        'TotalIdleTime(min)': 0.0,
        'TotalBusyTime(min)':0.0,
        'ProbabilityOfWaiting(Pwait)': 0.0,
        'ProbabilityOfBlocking(Pblock)': 1.0
    }