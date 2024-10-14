import pandas as pd
import mlrose_hiive as mlrose
import numpy as np
import matplotlib.pyplot as plt
from time import time
from timeit import default_timer as timer

max_iters = 5000
max_attempts = 500
random_seed = 42
np.random.seed(random_seed)

def run_optimization(problem, algorithm, *args, **kwargs):
    start = timer()
    best_state, best_fitness, fitness_curve = algorithm(problem, *args, **kwargs)
    elapsed_time = timer() - start
    return best_state, best_fitness, fitness_curve, elapsed_time

# Define random optimization problems
six_peaks = mlrose.DiscreteOpt(length=100, fitness_fn=mlrose.SixPeaks(t_pct=0.10), maximize=True)
knapsack = mlrose.DiscreteOpt(length=10, fitness_fn=mlrose.Knapsack(
    np.array([2, 10, 3, 5, 13, 7, 3, 11, 4, 16]), 
    np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 
    max_weight_pct=0.75), maximize=True)

# Run algorithms for Six Peaks
algorithms = {
    "RHC": mlrose.random_hill_climb,
    "SA": mlrose.simulated_annealing,
    "GA": mlrose.genetic_alg
}

fitness_curves = {}
eval_counts = {}
times = {}

for name, algo in algorithms.items():
    if name == "SA":
        schedule = mlrose.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001)
        best_state, best_fitness, fitness_curve, elapsed_time = run_optimization(six_peaks, algo, schedule=schedule,
                                                                                  max_attempts=max_attempts,
                                                                                  max_iters=max_iters,
                                                                                  curve=True, random_state=random_seed)
    else:
        best_state, best_fitness, fitness_curve, elapsed_time = run_optimization(six_peaks, algo,
                                                                                  max_attempts=max_attempts,
                                                                                  max_iters=max_iters,
                                                                                  curve=True, random_state=random_seed)
    fitness_curves[name] = (best_state, best_fitness, fitness_curve, elapsed_time)
    eval_counts[name] = len(fitness_curve)
    times[name] = elapsed_time

# Plotting fitness curves for Six Peaks
max_length = max(len(curve[2]) for curve in fitness_curves.values())
x = np.arange(1, max_length + 1)

# Prepare DataFrame for plotting
data = {
    'x': x,
    'RHC': np.concatenate((fitness_curves['RHC'][2][:, 0], np.full((max_length - len(fitness_curves['RHC'][2]),), np.nan))),
    'SA': np.concatenate((fitness_curves['SA'][2][:, 0], np.full((max_length - len(fitness_curves['SA'][2]),), np.nan))),
    'GA': np.concatenate((fitness_curves['GA'][2][:, 0], np.full((max_length - len(fitness_curves['GA'][2]),), np.nan)))
}
df = pd.DataFrame(data)

# Plot Fitness vs. Iterations
plt.figure(figsize=(10, 6))
for label, color in zip(['RHC', 'SA', 'GA'], ['blue', 'black', 'red']):
    plt.plot('x', label, data=df, color=color, linewidth=4, label=label)
plt.xlim(0, max_length)
plt.xlabel('Iterations')
plt.ylabel('Fitness Value')
plt.title('Six Peaks Optimization - Fitness vs. Iterations')
plt.legend()
plt.show()

# Run algorithms for Knapsack
fitness_curves_knapsack = {}
eval_counts_knapsack = {}
times_knapsack = {}

for name, algo in algorithms.items():
    if name == "SA":
        best_state, best_fitness, fitness_curve, elapsed_time = run_optimization(knapsack, algo, schedule=schedule,
                                                                                  max_attempts=max_attempts,
                                                                                  max_iters=max_iters,
                                                                                  curve=True, random_state=1)
    else:
        best_state, best_fitness, fitness_curve, elapsed_time = run_optimization(knapsack, algo,
                                                                                  max_attempts=max_attempts,
                                                                                  max_iters=max_iters,
                                                                                  curve=True, random_state=1)
    fitness_curves_knapsack[name] = (best_state, best_fitness, fitness_curve, elapsed_time)
    eval_counts_knapsack[name] = len(fitness_curve)
    times_knapsack[name] = elapsed_time

# Prepare DataFrame for plotting Knapsack
max_length_knapsack = max(len(curve[2]) for curve in fitness_curves_knapsack.values())
data_knapsack = {
    'x': np.arange(1, max_length_knapsack + 1),
    'RHC': np.concatenate((fitness_curves_knapsack['RHC'][2][:, 0], np.full((max_length_knapsack - len(fitness_curves_knapsack['RHC'][2]),), np.nan))),
    'SA': np.concatenate((fitness_curves_knapsack['SA'][2][:, 0], np.full((max_length_knapsack - len(fitness_curves_knapsack['SA'][2]),), np.nan))),
    'GA': np.concatenate((fitness_curves_knapsack['GA'][2][:, 0], np.full((max_length_knapsack - len(fitness_curves_knapsack['GA'][2]),), np.nan)))
}
df_knapsack = pd.DataFrame(data_knapsack)

# Plot results for Knapsack
plt.figure(figsize=(10, 6))
for label, color in zip(['RHC', 'SA', 'GA'], ['blue', 'black', 'red']):
    plt.plot('x', label, data=df_knapsack, color=color, linewidth=4, label=label)
plt.xlim(0, max_length_knapsack)
plt.xlabel('Iterations')
plt.ylabel('Fitness Value')
plt.title('Knapsack Optimization - Fitness vs. Iterations')
plt.legend()
plt.show()

# Plot Fitness vs. Problem Size
problem_sizes = [100, 10]
fitness_values = [fitness_curves['RHC'][1], fitness_curves_knapsack['RHC'][1]]
labels = ['Six Peaks', 'Knapsack']

plt.figure(figsize=(10, 6))
plt.bar(labels, fitness_values, color=['blue', 'orange'])
plt.xlabel('Problem Size')
plt.ylabel('Fitness Value')
plt.title('Fitness Values for Different Problem Sizes')
plt.show()

# Function Evaluations vs. Wall Clock Time
evaluations = list(eval_counts.values())
times_vals = list(times.values())

plt.figure(figsize=(10, 6))
plt.scatter(evaluations, times_vals, color='purple')
plt.title('Function Evaluations vs. Wall Clock Time')
plt.xlabel('Function Evaluations')
plt.ylabel('Wall Clock Time (seconds)')
for i, txt in enumerate(algorithms.keys()):
    plt.annotate(txt, (evaluations[i], times_vals[i]), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()
