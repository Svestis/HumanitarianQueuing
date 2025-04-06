"""
data_generator.py

Generates a synthetic dataset simulating individuals in a queue based on predefined JSON.
Each individual is assigned a unique ID, gender, age, transaction time, vulnerability score, and arrival time.
The dataset is saved to a CSV file and is intended for use in queuing simulations.

Last Updated: 2025_04_06
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Optional

# Mapping different configs for distributions
CONFIG_MAP = {
    'default': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.json')),
    'beta': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','configs', 'config_beta.json')),
    'triangular': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config_triangular.json')),
    'uniform': os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config_uniform.json'))
}

def load_config(config_type: Optional[str] = 'triangular') -> dict:
    """
    Load JSON config

    :param config_type: Type of config used
    :type config_type: str
    :return: A dictionary including the JSON contents
    :rtype: dict
    :raises FileNotFoundError: In case file has not being found for a given config
    """
    config_path: str = CONFIG_MAP.get(config_type, CONFIG_MAP['triangular']) # Load config with fallback

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, 'r') as file:
        return json.load(file)


def validate_config(config: dict) -> None:
    """
    Config basic schema validation
    :param config: A dictionary including the config values as loaded by the load_config function
    :type config: dict
    :return: None
    """
    assert 'num_records' in config
    assert abs(sum(config['columns']['gender']['probabilities']) - 1.0) < 1e-6, "Gender probabilities must sum to 1"

    required_columns: list = ['gender', 'age', 'transaction_time', 'vulnerability_score',
                        'start_time']

    for key in required_columns:
        assert key in config['columns'], f"Missing '{key}' in columns"

def ages(config: dict, num_records: int) -> np.ndarray:
    """
    Generate synthetic age values using a triangular distribution.
    :param config: Dictionary containing the params key with left, right and mode includes for the triangular distribution.
    :type config: dict
    :param num_records: Number of records
    :type num_records: int
    :return: Generated age values
    :rtype: np.ndarray
    """
    p = config['params']
    return np.random.triangular(left=p['left'], mode=p['mode'], right=p['right'], size=num_records).astype(int)

def transaction_time(config: dict, num_records:int) -> np.ndarray:
    """
    Generate synthetic transaction time values using a triangular distribution.
    :param config: Dictionary containing the params key with left, right and mode includes for the triangular distribution.
    :type config: dict
    :param num_records: Number of records
    :type num_records: int
    :return: Generated transaction_time values
    :rtype: np.ndarray
    """
    p = config['params']
    return np.random.triangular(left=p['left'], mode=p['mode'], right=p['right'], size=num_records).round(2)

def vulnerability(config: dict, num_records: int) -> np.ndarray:
    """
    Generate vulnerability scores using a beta distribution.
    :param config: Dictionary containing the params with a and b for the beta distribution.
    :type config: dict
    :param num_records: Number of records
    :type num_records: int
    :return: Vulnerability scores
    :rtype: np.ndarray
    """
    p = config['params']
    return np.round(np.random.beta(a=p['a'], b=p['b'], size=num_records), 2)

def gender(config: dict, num_records: int) -> np.ndarray:
    """
    Randomly sample gender values based on a given probability distribution.

    :param config: Dictionary with categories and propabilities
    :type config: dict
    :param num_records: Number of records
    :type num_records: int
    :return: Gender values
    :rtype: np.ndarray
    """
    return np.random.choice(config['values'], size=num_records, p=config['probabilities'])

def generate_arrival_offsets(distribution_cfg: dict, num_records: int) -> np.ndarray:
    """
    Generate arrival offsets based on config distrubution
    :param distribution_cfg: Distribution name from config file
    :type distribution_cfg: dict
    :param num_records: number of records from config
    :type num_records: int
    :return: an np.array with the implemented distribution
    :rtype: np.ndarray
    :raises ValueError: If the distribution that has been passed is not included in the if/else scenarios
    """
    dist = distribution_cfg.get("distribution", "triangular")
    params = distribution_cfg.get("params", {})

    if dist == "uniform":
        return np.random.uniform(
            low=params.get("low", 0),
            high=params.get("high", 600),
            size=num_records
        ).astype(int)
    elif dist == "triangular":
        return np.random.triangular(
            left=params.get("left", 0),
            mode=params.get("mode", 300),
            right=params.get("right", 600),
            size=num_records
        ).astype(int)
    elif dist == "beta":
        a = params.get("a", 2)
        b = params.get("b", 5)
        scale = params.get("scale", 600)
        return (np.random.beta(a, b, size=num_records) * scale).astype(int)
    else:
        raise ValueError(f"Unsupported distribution type: {dist}")


def data_generator(config: dict) -> pd.DataFrame:
    """
    Generates a synthetic dataset for queuing simulations.
    :param config: The specific configuration dictionary
    :type config: dict
    :return: The data in a dataframe
    :rtype: pd.DataFrame
    """
    validate_config(config)

    np.random.seed(42) # set random seed for reproducibility
    num_records: int = config['num_records']
    unique_ids: np.ndarray = np.arange(1, num_records + 1)

    # Gender distribution TODO: Follow a specific distribution?
    genders: np.ndarray = gender(config['columns']['gender'], num_records)

    # Age distribution TODO: Should change to another distribution for example normal?
    age: np.ndarray = ages(config['columns']['age'], num_records)

    # Transaction time in minutes TODO: Should just randomize?
    transaction_times: np.ndarray = transaction_time(config['columns']['transaction_time'], num_records)

    # Vulnerability score with beta distribution with low skewing TODO: Check vulnerability skewing based on literature
    vulnerability_scores: np.ndarray = vulnerability(config['columns']['vulnerability_score'], num_records)

    # Arrival times for 10 hours
    arrival_cfg: dict[str, str] = config.get("arrival_time", {"distribution": "triangular"}) # Load config with fallback
    arrival_offsets: np.ndarray = generate_arrival_offsets(arrival_cfg, num_records)
    start_time: datetime = datetime.strptime(config['columns']['start_time'], "%H:%M") # Convert to datetime

    # Add offsets to start time to define arrival time and changing back to HH:MM format
    arrival_times = [(start_time + timedelta(minutes=int(off))).strftime("%H:%M") for off in arrival_offsets]

    # Dataframe
    return pd.DataFrame({
        'ID': unique_ids,
        'Gender': genders,
        'Age': age,
        'TransactionTime': transaction_times,
        'VulnerabilityScore': vulnerability_scores,
        'ArrivalTime': arrival_times
    })


def save(config_type: Optional[str] = 'triangular',
         file_path: Optional[str] = "./src/resources/data_generator/synthetic_data.csv") -> pd.DataFrame:
    """
    Generate and save a synthetic dataset to a CSV file
    :param config_type: The key identifying which configuration file to load
    :type config_type: Optional[str]
    :param file_path: The file path where the CSV will be saved.
    :type file_path: Optional[str]
    :return: The generated synthetic dataset.
    :rtype: pd.DataFrame
    """
    config = load_config(config_type)
    df = data_generator(config)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return df

# TODO: Change to changeable JSON for different configs?
# TODO: Should return data instead of saving?
# TODO: Add different JSON configs for different scenarions?
