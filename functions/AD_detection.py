"Detection amd visualization functions"




import numpy as np
import math
from typing import Tuple, List, Dict
from matplotlib import pyplot as plt
from statistics import mean
import random
import itertools
from AD_QAOA_ext import AD_QAOA



def apply_circles_to_new_dataset(dataset: List[Tuple[int, float]], centers_and_radii: List[Tuple[Tuple[int, float], float]]) -> List[Tuple[int, float]]:
    """
    Identifies anomalies in a new dataset by checking if each point is within any defined circle of centers and radii.
    Adjust value (x) in tolerance (x * radius) if small adjustments for the sample inclusion/exclusion are required.
    
    Args:
        dataset (List[Tuple[int, float]]): The dataset to analyze, represented as a list of (timestamp, value) pairs.
        centers_and_radii (List[Tuple[Tuple[int, float], float]]): A list of circles, each defined by a center 
                                                                   (timestamp, value) and a radius.

    Returns:
        anomalies (List[Tuple[int, float]]): A list of (timestamp, value) pairs that are not covered by any circle, 
                                             indicating anomalies.
    """
    anomalies = []

    for timestamp, value in dataset:
        is_covered = False
        for (center, radius) in centers_and_radii:

            if isinstance(center, tuple) and len(center) == 2:
                center_timestamp, center_value = center
            else:
                continue

            distance = math.sqrt((center_timestamp - timestamp) ** 2 + (center_value - value) ** 2)
            tolerance = 0.0 * radius

            if distance <= radius + tolerance:
                is_covered = True
                break

        if not is_covered:
            anomalies.append((timestamp, value))

    print(f'Anomalies detected: {len(anomalies)}')
    print(f'Anomalies: {anomalies}')
    return anomalies




def plot_anomaly_detection_results(dataset: List[Tuple[int, float]], centers_and_radii: List[Tuple[Tuple[int, float], float]], anomalies: List[Tuple[int, float]], title: str):
    """
    Plots the dataset with anomalies highlighted and optional coverage circles.

    Args:
        dataset (List[Tuple[int, float]]): The dataset to plot, represented as a list of (timestamp, value) pairs.
        centers_and_radii (List[Tuple[Tuple[int, float], float]]): A list of circles defined by centers (timestamp, value)
                                                                   and their associated radii. Currently unused.
        anomalies (List[Tuple[int, float]]): A list of (timestamp, value) pairs identified as anomalies.
        title (str): The title of the plot.

    """
    fig, ax = plt.subplots(figsize=(20, 8))

    timestamps = [x[0] for x in dataset]
    values = [x[1] for x in dataset]
    ax.plot(timestamps, values, 'bo-', label='Dataset')

    anomaly_timestamps = [timestamp for (timestamp, _) in anomalies]
    anomaly_values = [value for (_, value) in anomalies]
    ax.plot(anomaly_timestamps, anomaly_values, 'ro', label='Anomalies')

    ax.set_title(title)
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Values")
    ax.legend()
    plt.grid(True)
    plt.show()




def plot_anomaly_detection_results_scaled(original_dataset: List[Tuple[int, float]],
                                          anomalies_scaled: List[Tuple[int, float]],
                                          title: str):
    """
    Plots the original dataset with anomalies highlighted, using the original scale for values.

    Args:
        original_dataset (List[Tuple[int, float]]): The original dataset represented as a list of (timestamp, value) pairs.
        anomalies_scaled (List[Tuple[int, float]]): A list of (timestamp, value) pairs identified as anomalies.
        title (str): The title of the plot.

    """
    fig, ax = plt.subplots(figsize=(16, 8))

    timestamps = [x[0] for x in original_dataset]
    values = [x[1] for x in original_dataset]
    ax.plot(timestamps, values, 'bo-', label='Original Dataset')

    anomaly_timestamps = [timestamp for (timestamp, _) in anomalies_scaled]
    anomaly_values = [original_dataset[timestamp][1] for timestamp in anomaly_timestamps]

    ax.plot(anomaly_timestamps, anomaly_values, 'ro', label='Anomalies')

    ax.set_title(title)
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Values")
    ax.legend()
    plt.grid(True)
    plt.show()




def plot_anomaly_detection_results_coverage(dataset: List[Tuple[int, float]], centers_and_radii: List[Tuple[Tuple[int, float], float]], anomalies: List[Tuple[int, float]], title: str):
    """
    Plots the dataset with coverage circles around centers, highlighting detected anomalies.

    Args:
        dataset (List[Tuple[int, float]]): The dataset to plot, represented as a list of (timestamp, value) pairs.
        centers_and_radii (List[Tuple[Tuple[int, float], float]]): A list of coverage circles, each defined by 
                                                                   a center (timestamp, value) and a radius.
        anomalies (List[Tuple[int, float]]): A list of (timestamp, value) pairs identified as anomalies.
        title (str): The title of the plot.

    """
    fig, ax = plt.subplots(figsize=(20, 8))

    timestamps = [x[0] for x in dataset]
    values = [x[1] for x in dataset]
    ax.plot(timestamps, values, 'bo-', label='Dataset')

    for i, (center, radius) in enumerate(centers_and_radii):
        circle = plt.Circle(center, radius, color='lightgreen', fill=True, linestyle='-', label='Coverage' if i == 0 else "")
        ax.add_artist(circle)
        ax.plot(center[0], center[1], 'yo', markersize=8, label='Centers' if i == 0 else "")

    anomaly_timestamps = [timestamp for (timestamp, _) in anomalies]
    anomaly_values = [value for (_, value) in anomalies]
    ax.plot(anomaly_timestamps, anomaly_values, 'ro', label='Anomalies')

    ax.set_title(title)
    ax.set_xlabel("Timestamps")
    ax.set_ylabel("Values")
    ax.legend()
    plt.grid(True)
    plt.show()
