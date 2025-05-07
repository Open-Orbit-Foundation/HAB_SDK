import time
import math
import numpy as np
import pandas as pd
from itertools import cycle
from functools import partial
import matplotlib.pyplot as plt
from utils import Integrator  # or from integrators import Integrator

# Damped harmonic oscillator
def oscillator(x, dx, k = 1, gamma = 0):
    return -k * x - gamma * dx

def analytical_solution(t, x0 = 1, dx0 = 0, k = 1, gamma = 0):
    delta = gamma ** 2 - 4 * k

    if delta < 0:  # Underdamped
        omega_d = np.sqrt(k - (gamma / 2) ** 2)
        A = x0
        B = (dx0 + (gamma / 2) * x0) / omega_d
        return np.exp(-gamma * t / 2) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    elif delta == 0:  # Critically damped
        A = x0
        B = dx0 + (gamma / 2) * x0
        return (A + B * t) * np.exp(- gamma * t / 2)

    else:  # Overdamped
        r1 = (- gamma + np.sqrt(delta)) / 2
        r2 = (- gamma - np.sqrt(delta)) / 2
        B = (dx0 - r1 * x0) / (r2 - r1)
        A = x0 - B
        return A * np.exp(r1 * t) + B * np.exp(r2 * t)

def rms_error(ref, test, max_threshold=1e6):
    delta = np.array(ref, dtype=np.float64) - np.array(test, dtype=np.float64)
    if np.any(np.isnan(delta)) or np.any(np.isinf(delta)) or np.max(np.abs(delta)) > max_threshold:
        return np.inf  # clearly diverged
    return np.sqrt(np.mean(delta ** 2))

def detect_turning_point(dt_values, errors, threshold_slope=3.0):
    """
    Identify the first dt where RMS error grows rapidly,
    based on slope of log-log curve.
    """
    log_dt = np.log10(dt_values)
    log_err = np.log10(errors)
    slope = np.gradient(log_err, log_dt)

    for i, s in enumerate(slope):
        if s > threshold_slope:
            return dt_values[i], i
    return None, None

def print_progress_bar(current, total, prefix='', suffix='', bar_length=100):
    fraction = current / total
    completed = int(bar_length * fraction)
    bar = '#' * completed + '-' * (bar_length - completed)
    print(f'\r{prefix} |{bar}| {current}/{total} {suffix}', end='', flush=True)

duration = 1_000.0  # total time (s)
x0, dx0 = 1.0, 0.0
k_base, gamma_base = 0.1, 0.3
methods = ['rk1_second_order', 'rk2_second_order', 'rk4_second_order']
time_steps = np.logspace(-2, 2, 100)  # dt from 0.01 to 10
error_threshold = 0.1

# Containers for results
errors = {m: [] for m in methods}
runtimes = {m: [] for m in methods}

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
plt.figure(figsize=(10, 6))
turning_points = []


for method in methods:

    integrator = getattr(Integrator, method)

    for i, dt in enumerate(time_steps):
        print_progress_bar(i + 1, len(time_steps), prefix=f"Benchmarking {method}...", suffix="Complete")

        steps = int(math.ceil(duration / dt))
        k = k_base / (dt * 2 * math.pi)
        gamma = gamma_base / (dt * 2 * math.pi)

        times = np.linspace(0, (steps - 1) * dt, steps, dtype=np.float32)
        exact = [analytical_solution(t, x0, dx0, k, gamma) for t in times]
        osc_func = partial(oscillator, k=k, gamma=gamma)

        x, dx = x0, dx0
        xs = np.empty(steps, dtype=np.float32)

        start = time.perf_counter()
        for i in range(steps):
            x, dx, _ = integrator(x, dx, osc_func, dt)
            xs[i] = x
        end = time.perf_counter()

        errors[method].append(rms_error(exact, xs))
        runtimes[method].append(end - start)

    print()  # newline after each method

    err_array = np.array(errors[method], dtype=np.float64)
    dt_array = np.array(time_steps, dtype=np.float64)

    # Mask only finite values
    mask = np.isfinite(err_array)
    dt_filtered = dt_array[mask]
    err_filtered = err_array[mask]

    i_thresh = next((i for i, err in enumerate(err_filtered) if err > error_threshold), len(err_filtered))
    turning_point, i_turn = detect_turning_point(dt_filtered, err_filtered)
    turning_points.append(turning_point)

    color = next(color_cycle)

    if turning_point:
        print(f"⚠️  {method} diverges at dt ≈ {turning_point:.4f}")
        if i_turn <= i_thresh:
            plt.plot(dt_filtered[:i_turn + 1], err_filtered[:i_turn + 1], label=method, color=color, linewidth=1.8)
        else:
            plt.plot(dt_filtered[:i_thresh], err_filtered[:i_thresh], label=method, color=color, linewidth=1.8)
        plt.axvline(turning_point, linestyle='--', color='gray', alpha=0.7)
        plt.text(turning_point, 0, f'dt≈{turning_point:.3f}', 
            rotation=90, va='top', ha='center', fontsize=8, color='gray',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray'))
        if i_thresh < len(dt_filtered):
            plt.plot(dt_filtered[i_thresh - 1:i_turn + 1], err_filtered[i_thresh - 1:i_turn + 1], linestyle='--', color=color, linewidth=1.8)
            print(f"⚠️  {method} exceeded threshold (error > {error_threshold}) at dt ≈ {dt_filtered[i_thresh]:.4f}")
        else:
            print(f"✅  {method} stayed below error threshold (≤ {error_threshold}) for all dt")
    else:
        print(f"✅  {method} remained stable over tested dt range")
        plt.plot(dt_filtered[:i_thresh], err_filtered[:i_thresh], label=method, color=color, linewidth=1.8)
        if i_thresh < len(dt_filtered):
            plt.plot(dt_filtered[i_thresh - 1:], err_filtered[i_thresh - 1:], linestyle='--', color=color, linewidth=1.8)
            print(f"⚠️  {method} exceeded threshold (error > {error_threshold}) at dt ≈ {dt_filtered[i_thresh]:.4f}")
        else:
            print(f"✅  {method} stayed below error threshold (≤ {error_threshold}) for all dt")

ymin, ymax = plt.ylim()
for turning_point in turning_points:
        if turning_point:
            xpos = turning_point
            ypos = ymax  # Use top of current plot area
            plt.text(xpos, ypos, f'dt≈{turning_point:.3f}',
                    rotation=90, va='top', ha='center', fontsize=8, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.5),
                    clip_on=True)
plt.axhline(error_threshold, linestyle='-', color='gray', alpha=0.9, label="_nolegend_")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Time Step (dt)")
plt.ylabel("RMS Error")
plt.title("Integrator RMS Error vs Time Step")
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.show()

# Convert to DataFrame for display or export
df = pd.DataFrame({
    'dt': time_steps,
    **{f'{m}_error': errors[m] for m in methods},
    **{f'{m}_time': runtimes[m] for m in methods},
})