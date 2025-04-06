"""
test_util.py

Validating metrics.py functionality

Last Updated: 2025_04_06
"""
import pandas as pd
from datetime import datetime
from helpers.models.util import build_processed_df, calculate_utilization

def test_build_processed_df() -> None:
    """
    Test that function is adding columns as needed
    :return: None
    """
    raw_df = pd.DataFrame({'ArrivalTime': [datetime(1900,1,1,8,0)]})
    indices = [0]
    start_times = [datetime(1900,1,1,8,0)]
    end_times = [datetime(1900,1,1,8,10)]
    server_ids = [0]
    processed = build_processed_df(raw_df, indices, start_times, end_times, server_ids)
    assert 'WaitTime' in processed.columns
    assert 'SystemTime' in processed.columns
    assert processed.iloc[0]['ServerID'] == 0

def test_utilization_calculation() -> None:
    """
    Ensure util is calculating correct utilization for known server usage
    :return: None
    """
    logs = [[(datetime(1900,1,1,8,0), datetime(1900,1,1,8,5))]]
    start = datetime(1900,1,1,8,0)
    end = datetime(1900,1,1,8,10)
    util_df = calculate_utilization(logs, start, end)
    assert not util_df.empty
    assert util_df['Utilization(%)'].iloc[0] == 50.0

def test_utilization_multiple_servers() -> None:
    """
    Test util metrics for different servers
    :return: None
    """
    logs = [
        [(datetime(1900,1,1,8,0), datetime(1900,1,1,8,5))],
        [(datetime(1900,1,1,8,2), datetime(1900,1,1,8,7))]
    ]
    start = datetime(1900,1,1,8,0)
    end = datetime(1900,1,1,8,10)
    df = calculate_utilization(logs, start, end)
    assert df.shape[0] == 2
    assert all(col in df.columns for col in ['Utilization(%)', 'IdleTime(min)', 'BusyTime(min)'])

def test_utilization_overlap_check() -> None:
    """
    Test overlap if servers are 100% occupied
    :return: None
    """
    logs = [
        [(datetime(1900,1,1,8,0), datetime(1900,1,1,8,10))],
        [(datetime(1900,1,1,8,0), datetime(1900,1,1,8,10))]
    ]
    df = calculate_utilization(logs, datetime(1900,1,1,8,0), datetime(1900,1,1,8,10))
    assert df['Utilization(%)'].iloc[0] == 100.0
    assert df['Utilization(%)'].iloc[1] == 100.0