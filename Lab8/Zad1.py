import keyboard as kb
import numpy as np
from dtw import dtw
import matplotlib.pyplot as plt

timestamp = 0.1


def dist(x, y):
    return np.abs(x - y)


def time_series_with_fixed_step(timings, timestamp=timestamp):
    timings = np.array(timings)
    time_grid = np.arange(timings.min(), timings.max(), timestamp)
    indices = np.zeros(time_grid.shape)
    for i, t in enumerate(time_grid):
        indices[i] = np.sum(timings <= t) - 1
    return time_grid, indices


def get_states_with_fixed_time(key_record):
    timings = [x.time - r1[0].time for x in key_record]
    events = [(x.event_type == 'down') * 2 - 1 for x in key_record]
    state = np.cumsum(events)
    time_grid, indices = time_series_with_fixed_step(timings)
    state_with_fixed_time = state[indices.astype(int)]
    return state_with_fixed_time


if __name__ == "__main__":
    print("Type: \"biometrics check\" and press q")
    r1 = kb.record("q")
    print("Type again: \"biometrics check\" and press q")
    r2 = kb.record("q")
    arr1 = get_states_with_fixed_time(r1)
    arr2 = get_states_with_fixed_time(r2)
    d, cost_matrix, acc_cost_matrix, path = dtw(arr1, arr2, dist=dist)
    plt.plot(path[0], path[1])
    plt.show()
    print(d)
