"""
test_fifo.py

Validating fifo.py functionality

Last Updated: 2025_04_06
"""
import pytest
import pandas as pd
from src import ros, mmc

@pytest.fixture
def test_data() -> pd.DataFrame:
    """
    Creating a sample df for testing.
    :return: A small dataframe for testing purposes
    :rtype: pd.Dataframe
    """
    data = {
        'ID': [1, 2, 3],
        'Gender': ['Male', 'Female', 'Other'],
        'Age': [25, 40, 60],
        'TransactionTime': [5, 10, 15],
        'VulnerabilityScore': [0.2, 0.5, 0.8],
        'ArrivalTime': ['08:00', '08:05', '08:10']
    }

    df: pd.DataFrame = pd.DataFrame(data)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format="%H:%M")
    return df

def test_structure(test_data: pd.DataFrame) -> None:
    """
    Testing if expected columns exist
    :param test_data: A dataframe with the test date to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    df_result, util_df, _, metrics = ros(original_df=test_data, num_servers=2, capacity=5)

    # Check required columns in result
    expected_cols = ['ID', 'Gender', 'Age', 'TransactionTime', 'VulnerabilityScore',
                     'ArrivalTime', 'StartTime', 'EndTime', 'WaitTime', 'SystemTime', 'ServerID']
    for col in expected_cols:
        assert col in df_result.columns

def test_metrics_keys(test_data: pd.DataFrame) -> None:
    """
    Testing if expected metrics exist
    :param test_data: A dataframe with the test data to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    _, _, _, metrics = ros(original_df=test_data, num_servers=2, capacity=5)

    # Check required metrics
    expected_keys = [
        'AverageWaitingTime(W)', 'AverageSystemTime(Ws)', 'Throughput', 'AverageQueueLength(Lq)',
        'AverageNumberInSystem(Ls)', 'AverageUtilization(%)', 'TotalIdleTime(min)', 'ProbabilityOfWaiting(Pwait)',
        'ProbabilityOfBlocking(Pblock)']

    for key in expected_keys:
        assert key in metrics

def test_no_nan_metrics(test_data: pd.DataFrame) -> None:
    """
    Ensuring no nan metrics
    :param test_data: A dataframe with the test data to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    _, _, _, metrics = ros(original_df=test_data, num_servers=2, capacity=5)

    for val in metrics.values():
        assert pd.notnull(val)

def test_utilization_structure(test_data: pd.DataFrame) -> None:
    """
    Testing if utilization data for mulitple runs exist
    :param test_data: A dataframe with the test data to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    _, util_df, _, _ = ros(original_df=test_data, num_servers=2, capacity=5)

    assert 'ServerID' in util_df.columns
    assert 'BusyTime(min)' in util_df.columns
    assert 'IdleTime(min)' in util_df.columns
    assert 'Utilization(%)' in util_df.columns

def test_all_blocked_() -> None:
    """Test all blocked
    :return: None"""
    data = {
        'ID': [1, 2],
        'Gender': ['M', 'F'],
        'Age': [30, 35],
        'TransactionTime': [10, 10],
        'VulnerabilityScore': [0.2, 0.3],
        'ArrivalTime': ['08:00', '08:01']
    }
    df = pd.DataFrame(data)
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M')

    processed, _, _, metrics = ros(original_df=df, num_servers=1, capacity=0)

    assert processed.empty
    assert metrics['ProbabilityOfBlocking(Pblock)'] == 1.0
    assert metrics['Throughput'] == 0.0

def test_no_waiting(test_data: pd.DataFrame):
    """
    Test that there is 0 waiting for known dataset
    :param test_data: A dataframe with the test data to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    processed, _, _, metrics = ros(original_df=test_data, num_servers=5, capacity=10)

    assert metrics['AverageWaitingTime(W)'] == 0.0
    assert metrics['ProbabilityOfWaiting(Pwait)'] == 0.0

def test_metrics() -> None:
    """
    Test utilization for known transaction and simulation time.
    :return: None
    """
    df = pd.DataFrame({
        'ID': [1],
        'Gender': ['F'],
        'Age': [35],
        'TransactionTime': [10],  # 10-minute transaction
        'VulnerabilityScore': [0.1],
        'ArrivalTime': ['08:00']
    })
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M')

    _, util_df, _,metrics = ros(original_df=df, num_servers=1, capacity=1)

    # Should be 100% utilization
    assert round(util_df.iloc[0]['Utilization(%)'], 2) == 100.0
    assert metrics['AverageUtilization(%)'] == 100.0
    assert metrics['AverageWaitingTime(W)'] == 0.0
    assert metrics['ProbabilityOfWaiting(Pwait)'] == 0.0


def test_metric_ranges() -> None:
    """
    Ensure metrics like probabilities and utilization are within expected bounds.
    :return: None
    """
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Gender': ['M', 'F', 'M'],
        'Age': [30, 25, 40],
        'TransactionTime': [5, 10, 15],
        'VulnerabilityScore': [0.3, 0.4, 0.5],
        'ArrivalTime': ['08:00', '08:05', '08:10']
    })
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M')

    _, _, _, metrics = ros(original_df=df, num_servers=2, capacity=3)

    assert 0.0 <= metrics['ProbabilityOfBlocking(Pblock)'] <= 1.0
    assert 0.0 <= metrics['ProbabilityOfWaiting(Pwait)'] <= 1.0
    assert 0.0 <= metrics['AverageUtilization(%)'] <= 100.0
    assert metrics['Throughput'] >= 0.0
    assert metrics['AverageWaitingTime(W)'] >= 0.0
    assert metrics['AverageSystemTime(Ws)'] >= 0.0
    assert metrics['AverageQueueLength(Lq)'] >= 0.0
    assert metrics['AverageNumberInSystem(Ls)'] >= 0.0

def test_empty_input() -> None:
    """Test behaviour does not break for empty input
    :return: None"""
    df = pd.DataFrame(columns=['ID', 'Gender', 'Age', 'TransactionTime', 'VulnerabilityScore', 'ArrivalTime'])
    processed, util_df, _, metrics = ros(original_df=df, num_servers=1)

    assert processed.empty
    assert util_df.empty
    assert metrics['Throughput'] == 0.0

def test_partial_blocking() -> None:
    """
    Test that blocking works for known capacity
    :return: None
    """
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Gender': ['M', 'F', 'M'],
        'Age': [30, 40, 25],
        'TransactionTime': [10, 10, 10],
        'VulnerabilityScore': [0.1, 0.2, 0.3],
        'ArrivalTime': ['08:00', '08:01', '08:02']
    })
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M')

    processed, _, _, metrics = ros(original_df=df, num_servers=1, capacity=2)

    assert not processed.empty
    assert 0 <= metrics['ProbabilityOfBlocking(Pblock)'] < 1

def test_simultaneous_arrivals() -> None:
    """Test behaviour for customers arriving at the same time
    :return: None"""
    df = pd.DataFrame({
        'ID': [1, 2, 3],
        'Gender': ['M', 'F', 'M'],
        'Age': [30, 35, 40],
        'TransactionTime': [5, 10, 15],
        'VulnerabilityScore': [0.2, 0.3, 0.4],
        'ArrivalTime': ['08:00', '08:00', '08:00']
    })
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M')
    processed, _, _, metrics = ros(original_df=df, num_servers=2, capacity=3)
    assert not processed.empty
    assert metrics['ProbabilityOfWaiting(Pwait)'] > 0

def test_output_types(test_data: pd.DataFrame) -> None:
    """Testing output types conform to the expected outputs for returned data

    :param test_data: A dataframe with the test data to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    processed, util_df, server_logs, metrics = ros(original_df=test_data, num_servers=2)
    assert isinstance(processed, pd.DataFrame)
    assert isinstance(util_df, pd.DataFrame)
    assert isinstance(server_logs, list)
    assert isinstance(metrics, dict)

def test_ros_order(test_data: pd.DataFrame) -> None:
    """
    Test that ROS does not always serve in mmc order.
    :param test_data: A dataframe with the test data to be used
    :type test_data: pd.DataFrame
    :return: None
    """
    first_run, *_ = ros(original_df=test_data.copy(), num_servers=1)
    second_run, *_ = ros(original_df=test_data.copy(), num_servers=5)
    assert not first_run['StartTime'].equals(second_run['StartTime']), "ROS behaves like FIFO; should be random"