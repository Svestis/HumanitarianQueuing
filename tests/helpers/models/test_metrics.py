"""
test_metrics.py

Validating metrics.py functionality

Last Updated: 2025_04_06
"""
import pandas as pd
from datetime import datetime
from src.helpers.models.metrics import calculate_metrics

def test_metrics_with_valid_data() -> None:
    """
    The metric calculations with known input
    :return: None
    """
    # Brute force values
    df = pd.DataFrame({
        'StartTime': [datetime(1900, 1, 1, 8, 0), datetime(1900, 1, 1, 8, 5)],
        'EndTime': [datetime(1900, 1, 1, 8, 10), datetime(1900, 1, 1, 8, 15)],
        'ArrivalTime': [datetime(1900, 1, 1, 8, 0), datetime(1900, 1, 1, 8, 4)],
        'TransactionTime': [10, 10],
        'WaitTime': [0, 1],
        'SystemTime': [10, 11]
    })

    # Brute force values
    server_logs = [[(datetime(1900, 1, 1, 8, 0), datetime(1900, 1, 1, 8, 10))],
                   [(datetime(1900, 1, 1, 8, 5), datetime(1900, 1, 1, 8, 15))]]

    metrics = calculate_metrics(
        df, server_logs, blocked_count=0, total_arrivals=2,
        simulation_start=df['ArrivalTime'].min(),
        simulation_end=df['EndTime'].max()
    )
    assert isinstance(metrics, dict)
    assert metrics['AverageWaitingTime(W)'] >= 0
    assert metrics['ProbabilityOfBlocking(Pblock)'] == 0.0

def test_metrics_all_blocked() -> None:
    """
    Test metrics when all arrivals are blocked
    :return: None
    """
    df = pd.DataFrame(columns=['StartTime', 'EndTime', 'ArrivalTime', 'TransactionTime', 'WaitTime', 'SystemTime'])
    server_logs = [[]]
    metrics = calculate_metrics(
        df, server_logs, blocked_count=2, total_arrivals=2,
        simulation_start=datetime(1900,1,1,8,0),
        simulation_end=datetime(1900,1,1,8,10)
    )
    assert metrics['ProbabilityOfBlocking(Pblock)'] == 1.0
    assert metrics['Throughput'] == 0.0


def test_metrics_zero_duration() -> None:
    """
    Test metrics for a zero duration
    :return: None
    """
    df = pd.DataFrame({
        'StartTime': [datetime(1900, 1, 1, 8, 0)],
        'EndTime': [datetime(1900, 1, 1, 8, 0)],
        'ArrivalTime': [datetime(1900, 1, 1, 8, 0)],
        'TransactionTime': [0],
        'WaitTime': [0],
        'SystemTime': [0]
    })
    server_logs = [[(datetime(1900, 1, 1, 8, 0), datetime(1900, 1, 1, 8, 0))]]
    metrics = calculate_metrics(df, server_logs, blocked_count=0, total_arrivals=1,
                                 simulation_start=datetime(1900, 1, 1, 8, 0),
                                 simulation_end=datetime(1900, 1, 1, 8, 0))
    assert metrics['Throughput'] == 1_000_000.0
