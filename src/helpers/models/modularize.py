"""
modularize.py

Modularize commonly used code chunks

Last Updated: 2025_04_06
"""

from datetime import timedelta

def server_assignments(row, sid, server_end_times, server_logs):
    arrival_time = row['ArrivalTime']
    start_time = max(arrival_time, server_end_times[sid])
    end_time = start_time + timedelta(minutes=int(row['TransactionTime']))
    server_end_times[sid] = end_time
    server_logs[sid].append((start_time, end_time))
    return start_time, end_time

# TODO: Implement modulirization