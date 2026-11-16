import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def get_states(k: int) -> list[tuple[int, int]]:
    """
    Returns a list of all possible states in the CTMC.
    Each state is represented as a tuple (n, j), where n is the number of customers in the system
    and j is the current phase of the process.
    """
    return [(n,j) for j in range(2) for n in range(k+1)]

def build_queue_generator(k: int, arrival_rate: float, mu: float) -> np.ndarray:
    """
    This function build our single generator matrix for the CTMC individually either T0 or T1 individually for the build_mmpp_generator() function to use. The generated matrix is a square matrix of size (k+1) x (k+1).
    """
    matrix = np.zeros((k+1, k+1))

    np.fill_diagonal(matrix, -(arrival_rate + mu))

    for i in range(k+1):
        for j in range(k+1):
            if i == j:
                continue
            elif j == i + 1 and i < k:
                matrix[i][j] = arrival_rate
            elif j == i - 1 and i > 0:
                matrix[i][j] = mu 
    matrix[0][0] = -arrival_rate
    matrix[k][k] = -mu

    return matrix

def build_mmpp_generator(k: int, lambda_0: float, lambda_1: float, mu: float, alpha: float, beta: float) -> np.ndarray:
    """
    This function builds the generator matrix for a Markov Modulated Poisson Process (MMPP) queueing system.
    """
    t0 = build_queue_generator(k, lambda_0, mu)
    t1 = build_queue_generator(k, lambda_1, mu)
    I = np.eye(k+1)
    matrix = np.block([[t0 - alpha * I, alpha * I], [beta * I, t1 - beta * I]])
    return matrix

def validate_generator(Q: np.ndarray, k: int) -> str:
    """
    Validates the generator matrix Q for a Markov Modulated Poisson Process (MMPP) queueing system.
    """
    if Q.shape != (2*(k+1), 2*(k+1)):
        return "Error: The generator matrix Q must be of size (2*(k+1)) x (2*(k+1))."

    if not np.allclose(Q.sum(axis=1), 0):
        return "Error: The rows of the generator matrix Q must sum to zero."

    if not np.all(np.diag(Q) <= 0):
        return "Error: The diagonal elements of the generator matrix Q must be non-positive."

    off_diagonal_mask = ~np.eye(Q.shape[0], dtype=bool)
    if not np.all(Q[off_diagonal_mask] >= 0):
        return "Error: All off-diagonal entries must be non-negative."

    if not np.isfinite(Q).all():
        return "Error: The generator matrix Q must not contain any infinite or NaN values."
    
    return "Validation successful: The generator matrix Q is valid."

def solve_forward_equation(Q, p_initial, time_points):
    """
    Solves the Kolmogorov Forward Equation:

        dp(t)/dt = p(t) Q

    Q           : Generator matrix
    p_initial   : Initial probability distribution
    time_points : Time values at which probabilities are calculated

    Returns:
        Probability distribution at each time point.
    """

    Q = np.asarray(Q, dtype=float)
    p_initial = np.asarray(p_initial, dtype=float)
    time_points = np.asarray(time_points, dtype=float)

    # Check generator matrix dimensions
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Q must be a square matrix.")

    n = Q.shape[0]

    # Check initial probability vector
    if p_initial.shape != (n,):
        raise ValueError("Initial probability vector has incorrect size.")

    if np.any(p_initial < 0) or not np.isclose(p_initial.sum(), 1):
        raise ValueError("Initial probabilities must be non-negative and sum to 1.")

    # Check generator properties
    if not np.all(np.isfinite(Q)):
        raise ValueError("Q contains non-finite values.")

    if not np.allclose(Q.sum(axis=1), 0):
        raise ValueError("Rows of Q must sum to zero.")

    # Check time points
    if time_points.ndim != 1 or len(time_points) < 2:
        raise ValueError("Provide at least two time points.")

    if time_points[0] < 0 or np.any(np.diff(time_points) <= 0):
        raise ValueError("Time points must be strictly increasing and non-negative.")

    # Column-vector form:
    # dp/dt = Q.T @ p

    def forward_equation(t, p):
        return Q.T @ p

    solution = solve_ivp(
        forward_equation,
        t_span=(time_points[0], time_points[-1]),
        y0=p_initial,
        t_eval=time_points,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    # Convert solution into rows:
    # Each row represents probabilities at one time point.

    probabilities = solution.y.T

    return probabilities


k =int(input("Enter the maximum number of customers in the system (k): "))
lambda_0 = float(input("Enter the arrival rate in state 0 (lambda_0): "))
lambda_1 = float(input("Enter the arrival rate in state 1 (lambda_1): "))
mu = float(input("Enter the service rate (mu): "))
alpha = float(input("Enter the transition rate from state 0 to state 1 (alpha): "))
beta = float(input("Enter the transition rate from state 1 to state 0 (beta): "))
Q = build_mmpp_generator(k, lambda_0, lambda_1, mu, alpha, beta)
print("Q \n", Q)
p_initial = np.zeros(2 * (k + 1))
p_initial[0] = 1.0
time_points = np.linspace(0, 10, 201)

probabilities = solve_forward_equation(
    Q,
    p_initial,
    time_points
)


validation_result = validate_generator(Q, k)
print(validation_result)

# ==========================================
# SOLVE THE FORWARD EQUATION
# ==========================================

probabilities = solve_forward_equation(
    Q,
    p_initial,
    time_points
)


# ==========================================
# DISPLAY NUMERICAL RESULTS
# ==========================================

print("\nKolmogorov Forward Equation Results")

print("\nInitial Probability Distribution:")
print(probabilities[0])

print("\nProbability Distribution at Final Time:")
print(probabilities[-1])

print("\nTotal Probability at Final Time:")
print(probabilities[-1].sum())

print("\nMinimum Calculated Probability:")
print(probabilities.min())

# Check probability conservation at every time point.
total_probabilities = probabilities.sum(axis=1)

print("\nMaximum Probability Conservation Error:")
print(np.max(np.abs(total_probabilities - 1)))

print("\nForward Equation Solved Successfully!")



# ==========================================
# PLOT 1: ALL COMBINED STATE PROBABILITIES
# ==========================================

states = get_states(k)

plt.figure(figsize=(12, 6))

for i, state in enumerate(states):
    plt.plot(
        time_points,
        probabilities[:, i],
        label=f"State {state}"
    )

plt.title("Transient State Probabilities of MMPP Queue")
plt.xlabel("Time")
plt.ylabel("State Probability")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("C:\\Workspace\\active-projects\\stochastic-queueing-model\\app\\models\\komolove_equation_plots\\plot_1_combined_states.png", dpi=300, bbox_inches="tight")
plt.show()


# ==========================================
# PLOT 2: QUEUE-LENGTH PROBABILITIES
# ==========================================

queue_probabilities = np.zeros((len(time_points), k + 1))

for i in range(k + 1):

    # Normal regime probability
    normal_probability = probabilities[:, i]

    # Peak regime probability
    peak_probability = probabilities[:, (k + 1) + i]

    # Combined queue-length probability
    queue_probabilities[:, i] = (
        normal_probability + peak_probability
    )


plt.figure(figsize=(12, 6))

for i in range(k + 1):
    plt.plot(
        time_points,
        queue_probabilities[:, i],
        label=f"Queue length = {i}"
    )

plt.title("Queue-Length Probabilities Over Time")
plt.xlabel("Time")
plt.ylabel("Probability")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("C:\\Workspace\\active-projects\\stochastic-queueing-model\\app\\models\\komolove_equation_plots\\plot_2_combined_states.png", dpi=300, bbox_inches="tight")
plt.show()



# ==========================================
# PLOT 3: TRAFFIC REGIME PROBABILITIES
# ==========================================

normal_regime_probability = probabilities[:, :k+1].sum(axis=1)

peak_regime_probability = probabilities[:, k+1:].sum(axis=1)


plt.figure(figsize=(12, 6))

plt.plot(
    time_points,
    normal_regime_probability,
    label="Normal Traffic",
    linewidth=2
)

plt.plot(
    time_points,
    peak_regime_probability,
    label="Peak Traffic",
    linewidth=2
)

plt.title("Normal and Peak Traffic Regime Probabilities")
plt.xlabel("Time")
plt.ylabel("Probability")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("C:\\Workspace\\active-projects\\stochastic-queueing-model\\app\\models\\komolove_equation_plots\\plot_3_combined_states.png", dpi=300, bbox_inches="tight")
plt.show()



# ==========================================
# PLOT 4: FULL CAPACITY PROBABILITY
# ==========================================

full_capacity_probability = (
    probabilities[:, k] +
    probabilities[:, 2*k + 1]
)


plt.figure(figsize=(12, 6))

plt.plot(
    time_points,
    full_capacity_probability,
    label="Probability of Full Capacity",
    linewidth=2
)

plt.title("Probability of Queue Reaching Full Capacity")
plt.xlabel("Time")
plt.ylabel("Probability")

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("C:\\Workspace\\active-projects\\stochastic-queueing-model\\app\\models\\komolove_equation_plots\\plot_4_combined_states.png", dpi=300, bbox_inches="tight")
plt.show()



# ============================================================
# PART 6: FIRST-PASSAGE TIME ANALYSIS FOR THE MMPP QUEUE
# ============================================================

def build_transient_generator(Q, k):
    """
    Construct the transient subgenerator Q_T.

    Target states:
        (k, 0) and (k, 1)

    Transient states:
        (0, 0), ..., (k-1, 0),
        (0, 1), ..., (k-1, 1)

    Original diagonal entries are retained.
    """

    transient_indices = [
        j * (k + 1) + n
        for j in range(2)
        for n in range(k)
    ]

    Q_T = Q[np.ix_(transient_indices, transient_indices)]

    return Q_T


def calculate_expected_first_passage_times(Q_T):
    """
    Solve Q_T m = -1.

    Returns expected first-passage times for every
    transient initial state.
    """

    number_of_states = Q_T.shape[0]

    ones = np.ones(number_of_states)

    expected_times = np.linalg.solve(-Q_T, ones)

    return expected_times


def calculate_first_passage_distribution(
    Q_T,
    initial_state_index,
    time_points
):
    """
    Calculate:
        Survival probability S(t)
        First-passage CDF F(t)
        First-passage density f(t)

    initial_state_index is the index in Q_T.
    """

    number_of_states = Q_T.shape[0]

    p_initial_transient = np.zeros(number_of_states)
    p_initial_transient[initial_state_index] = 1.0

    # Rates of transitions from transient states
    # into either full-capacity target state.
    exit_rates = -Q_T @ np.ones(number_of_states)

    def transient_forward_equation(t, p):
        return Q_T.T @ p

    solution = solve_ivp(
        transient_forward_equation,
        (time_points[0], time_points[-1]),
        p_initial_transient,
        t_eval=time_points,
        method="DOP853",
        rtol=1e-9,
        atol=1e-11
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    transient_probabilities = solution.y.T

    # Probability that capacity has not yet been reached.
    survival_probability = transient_probabilities.sum(axis=1)

    # Probability that capacity has been reached at least once.
    first_passage_cdf = 1.0 - survival_probability

    # First-passage probability density.
    first_passage_density = transient_probabilities @ exit_rates

    return (
        survival_probability,
        first_passage_cdf,
        first_passage_density
    )


# ------------------------------------------------------------
# 6.1 Construct transient generator
# ------------------------------------------------------------

Q_T = build_transient_generator(Q, k)

print("\n===================================")
print("FIRST-PASSAGE TIME ANALYSIS")
print("===================================")

print("\nTransient Generator Matrix Q_T:")
print(Q_T)

print("\nDimension of Q_T:", Q_T.shape)


# ------------------------------------------------------------
# 6.2 Expected first-passage times
# ------------------------------------------------------------

expected_times = calculate_expected_first_passage_times(Q_T)

# State ordering in Q_T:
# (0,0), (1,0), ..., (k-1,0),
# (0,1), (1,1), ..., (k-1,1)

normal_expected_times = expected_times[:k]
peak_expected_times = expected_times[k:]

print("\nExpected First-Passage Times")
print("-----------------------------------")

print("Initial Queue Length | Normal | Peak")

for i in range(k):
    print(
        f"{i:20d} | "
        f"{normal_expected_times[i]:.6f} | "
        f"{peak_expected_times[i]:.6f}"
    )

print("\nExpected time from empty queue under normal traffic:")
print(normal_expected_times[0])

print("\nExpected time from empty queue under peak traffic:")
print(peak_expected_times[0])


# ------------------------------------------------------------
# 6.3 First-passage distribution from empty queue
# ------------------------------------------------------------

# Q_T index 0 = (0,0)
# Q_T index k = (0,1)

(
    survival_normal,
    cdf_normal,
    density_normal
) = calculate_first_passage_distribution(
    Q_T,
    0,
    time_points
)

(
    survival_peak,
    cdf_peak,
    density_peak
) = calculate_first_passage_distribution(
    Q_T,
    k,
    time_points
)


print("\nFirst-Passage Probabilities at Final Time")
print("-----------------------------------")

print("Normal initial regime:")
print("Survival probability:", survival_normal[-1])
print("First-passage CDF:", cdf_normal[-1])

print("\nPeak initial regime:")
print("Survival probability:", survival_peak[-1])
print("First-passage CDF:", cdf_peak[-1])


# ============================================================
# PLOT 5: FIRST-PASSAGE CUMULATIVE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_points,
    cdf_normal,
    label="Initially Normal Traffic",
    linewidth=2
)

plt.plot(
    time_points,
    cdf_peak,
    label="Initially Peak Traffic",
    linewidth=2
)

plt.title("First-Passage Cumulative Distribution")
plt.xlabel("Time")
plt.ylabel("Probability of Reaching Full Capacity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_5_first_passage_cdf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 6: FIRST-PASSAGE PROBABILITY DENSITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_points,
    density_normal,
    label="Initially Normal Traffic",
    linewidth=2
)

plt.plot(
    time_points,
    density_peak,
    label="Initially Peak Traffic",
    linewidth=2
)

plt.title("First-Passage Time Probability Density")
plt.xlabel("Time")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_6_first_passage_density.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 7: SURVIVAL PROBABILITY
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    time_points,
    survival_normal,
    label="Initially Normal Traffic",
    linewidth=2
)

plt.plot(
    time_points,
    survival_peak,
    label="Initially Peak Traffic",
    linewidth=2
)

plt.title("First-Passage Survival Probability")
plt.xlabel("Time")
plt.ylabel("Probability of Not Yet Reaching Capacity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_7_survival_probability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 8: EXPECTED FIRST-PASSAGE TIME BY INITIAL QUEUE LENGTH
# ============================================================

initial_queue_lengths = np.arange(k)

plt.figure(figsize=(10, 6))

plt.plot(
    initial_queue_lengths,
    normal_expected_times,
    marker="o",
    label="Initially Normal Traffic",
    linewidth=2
)

plt.plot(
    initial_queue_lengths,
    peak_expected_times,
    marker="s",
    label="Initially Peak Traffic",
    linewidth=2
)

plt.title("Expected First-Passage Time vs Initial Queue Length")
plt.xlabel("Initial Queue Length")
plt.ylabel("Expected Time to Reach Full Capacity")
plt.xticks(initial_queue_lengths)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_8_expected_first_passage.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PART 7: EXPERIMENT 2 - EFFECT OF SYSTEM CAPACITY
# ============================================================

capacities = [3, 5, 10, 15]

capacity_results_normal = []
capacity_results_peak = []

for capacity in capacities:

    Q_capacity = build_mmpp_generator(
        capacity,
        lambda_0,
        lambda_1,
        mu,
        alpha,
        beta
    )

    Q_T_capacity = build_transient_generator(
        Q_capacity,
        capacity
    )

    expected_capacity = calculate_expected_first_passage_times(
        Q_T_capacity
    )

    # Initially empty, normal regime
    capacity_results_normal.append(expected_capacity[0])

    # Initially empty, peak regime
    capacity_results_peak.append(expected_capacity[capacity])


print("\n===================================")
print("EXPERIMENT 2: EFFECT OF CAPACITY")
print("===================================")

print("Capacity | Normal Initial | Peak Initial")

for i, capacity in enumerate(capacities):
    print(
        f"{capacity:8d} | "
        f"{capacity_results_normal[i]:14.6f} | "
        f"{capacity_results_peak[i]:12.6f}"
    )


# Plot expected first-passage time against capacity

plt.figure(figsize=(10, 6))

plt.plot(
    capacities,
    capacity_results_normal,
    marker="o",
    label="Initially Normal Traffic",
    linewidth=2
)

plt.plot(
    capacities,
    capacity_results_peak,
    marker="s",
    label="Initially Peak Traffic",
    linewidth=2
)

plt.title("Effect of System Capacity on Expected First-Passage Time")
plt.xlabel("System Capacity (K)")
plt.ylabel("Expected First-Passage Time")
plt.xticks(capacities)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_9_capacity_experiment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PART 8: EXPERIMENT 3 - EFFECT OF PEAK ARRIVAL RATE
# ============================================================

peak_arrival_rates = [3, 4, 5, 6, 7, 8]

arrival_results_normal = []
arrival_results_peak = []

for peak_rate in peak_arrival_rates:

    Q_experiment = build_mmpp_generator(
        k,
        lambda_0,
        peak_rate,
        mu,
        alpha,
        beta
    )

    Q_T_experiment = build_transient_generator(
        Q_experiment,
        k
    )

    expected_experiment = calculate_expected_first_passage_times(
        Q_T_experiment
    )

    arrival_results_normal.append(expected_experiment[0])
    arrival_results_peak.append(expected_experiment[k])


print("\n===================================")
print("EXPERIMENT 3: EFFECT OF PEAK ARRIVAL RATE")
print("===================================")

print("Peak Arrival Rate | Normal Initial | Peak Initial")

for i, peak_rate in enumerate(peak_arrival_rates):
    print(
        f"{peak_rate:17.2f} | "
        f"{arrival_results_normal[i]:14.6f} | "
        f"{arrival_results_peak[i]:12.6f}"
    )


# Plot expected first-passage time against peak arrival rate

plt.figure(figsize=(10, 6))

plt.plot(
    peak_arrival_rates,
    arrival_results_normal,
    marker="o",
    label="Initially Normal Traffic",
    linewidth=2
)

plt.plot(
    peak_arrival_rates,
    arrival_results_peak,
    marker="s",
    label="Initially Peak Traffic",
    linewidth=2
)

plt.title("Effect of Peak Arrival Rate on Expected First-Passage Time")
plt.xlabel("Peak Arrival Rate (lambda_1)")
plt.ylabel("Expected First-Passage Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_10_peak_arrival_experiment.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PART 9: SAVE EXPERIMENT RESULTS
# ============================================================

capacity_table = np.column_stack((
    capacities,
    capacity_results_normal,
    capacity_results_peak
))

np.savetxt(
    "experiment_capacity_results.csv",
    capacity_table,
    delimiter=",",
    header="K,Expected_Time_Normal,Expected_Time_Peak",
    comments="",
    fmt="%.6f"
)

arrival_table = np.column_stack((
    peak_arrival_rates,
    arrival_results_normal,
    arrival_results_peak
))

np.savetxt(
    "experiment_peak_arrival_results.csv",
    arrival_table,
    delimiter=",",
    header="Lambda_1,Expected_Time_Normal,Expected_Time_Peak",
    comments="",
    fmt="%.6f"
)

print("\nExperiment results saved successfully.")


# ============================================================
# PART 10: ADDITIONAL PARAMETER SENSITIVITY EXPERIMENTS
# ============================================================

# Baseline parameters
baseline = {
    "K": k,
    "lambda_0": lambda_0,
    "lambda_1": lambda_1,
    "mu": mu,
    "alpha": alpha,
    "beta": beta
}

# Additional experimental scenarios
scenarios = {
    "Baseline": baseline,

    "Higher Arrivals": {
        "K": 5,
        "lambda_0": 3,
        "lambda_1": 7,
        "mu": 4,
        "alpha": 0.8,
        "beta": 0.3
    },

    "Higher Service Rate": {
        "K": 5,
        "lambda_0": 2,
        "lambda_1": 5,
        "mu": 6,
        "alpha": 0.8,
        "beta": 0.3
    },

    "More Frequent Peak": {
        "K": 5,
        "lambda_0": 2,
        "lambda_1": 5,
        "mu": 4,
        "alpha": 1.6,
        "beta": 0.3
    },

    "Longer Peak Period": {
        "K": 5,
        "lambda_0": 2,
        "lambda_1": 5,
        "mu": 4,
        "alpha": 0.8,
        "beta": 0.15
    },

    "Larger Capacity": {
        "K": 10,
        "lambda_0": 3,
        "lambda_1": 7,
        "mu": 4,
        "alpha": 0.8,
        "beta": 0.3
    }
}


# ------------------------------------------------------------
# 10.1 Calculate expected first-passage times
# ------------------------------------------------------------

experiment_results = {}

for scenario_name, params in scenarios.items():

    scenario_Q = build_mmpp_generator(
        params["K"],
        params["lambda_0"],
        params["lambda_1"],
        params["mu"],
        params["alpha"],
        params["beta"]
    )

    scenario_Q_T = build_transient_generator(
        scenario_Q,
        params["K"]
    )

    scenario_expected_times = (
        calculate_expected_first_passage_times(
            scenario_Q_T
        )
    )

    # Empty queue, normal regime
    normal_time = scenario_expected_times[0]

    # Empty queue, peak regime
    peak_time = scenario_expected_times[params["K"]]

    experiment_results[scenario_name] = {
        "normal": normal_time,
        "peak": peak_time,
        "parameters": params
    }


# ------------------------------------------------------------
# 10.2 Display comparison table
# ------------------------------------------------------------

print("\n==============================================")
print("ADDITIONAL PARAMETER SENSITIVITY EXPERIMENTS")
print("==============================================")

print(
    f"{'Scenario':25s} "
    f"{'Normal Initial':>16s} "
    f"{'Peak Initial':>16s}"
)

for name, result in experiment_results.items():

    print(
        f"{name:25s} "
        f"{result['normal']:16.6f} "
        f"{result['peak']:16.6f}"
    )


# ------------------------------------------------------------
# 10.3 Calculate percentage changes from baseline
# ------------------------------------------------------------

baseline_normal = experiment_results["Baseline"]["normal"]
baseline_peak = experiment_results["Baseline"]["peak"]

print("\nPercentage Change from Baseline")
print("----------------------------------------------")

print(
    f"{'Scenario':25s} "
    f"{'Normal Change':>16s} "
    f"{'Peak Change':>16s}"
)

for name, result in experiment_results.items():

    normal_change = (
        (result["normal"] - baseline_normal)
        / baseline_normal
    ) * 100

    peak_change = (
        (result["peak"] - baseline_peak)
        / baseline_peak
    ) * 100

    print(
        f"{name:25s} "
        f"{normal_change:15.2f}% "
        f"{peak_change:15.2f}%"
    )


# ------------------------------------------------------------
# 10.4 Save experiment results to CSV
# ------------------------------------------------------------

scenario_table = []

for name, result in experiment_results.items():

    params = result["parameters"]

    scenario_table.append([
        name,
        params["K"],
        params["lambda_0"],
        params["lambda_1"],
        params["mu"],
        params["alpha"],
        params["beta"],
        result["normal"],
        result["peak"]
    ])

np.savetxt(
    "experiment_sensitivity_results.csv",
    np.array(scenario_table, dtype=object),
    delimiter=",",
    fmt="%s",
    header=(
        "Scenario,K,Lambda_0,Lambda_1,Mu,Alpha,Beta,"
        "Expected_Time_Normal,Expected_Time_Peak"
    ),
    comments=""
)

print("\nSensitivity results saved successfully.")


# ============================================================
# PLOT 11: EXPECTED FIRST-PASSAGE TIME COMPARISON
# ============================================================

scenario_names = list(experiment_results.keys())

normal_values = [
    experiment_results[name]["normal"]
    for name in scenario_names
]

peak_values = [
    experiment_results[name]["peak"]
    for name in scenario_names
]

x = np.arange(len(scenario_names))
width = 0.35

plt.figure(figsize=(12, 6))

plt.bar(
    x - width / 2,
    normal_values,
    width,
    label="Initially Normal Traffic"
)

plt.bar(
    x + width / 2,
    peak_values,
    width,
    label="Initially Peak Traffic"
)

plt.xticks(
    x,
    scenario_names,
    rotation=20,
    ha="right"
)

plt.title("Expected First-Passage Time Across Scenarios")
plt.xlabel("Experimental Scenario")
plt.ylabel("Expected First-Passage Time")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_11_sensitivity_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PLOT 12: FIRST-PASSAGE CDF COMPARISON
# BASELINE VS HIGHER ARRIVAL AND SERVICE RATES
# ============================================================

cdf_scenarios = [
    "Baseline",
    "Higher Arrivals",
    "Higher Service Rate"
]

comparison_times = np.linspace(0, 20, 401)

plt.figure(figsize=(12, 6))

for scenario_name in cdf_scenarios:

    params = scenarios[scenario_name]

    scenario_Q = build_mmpp_generator(
        params["K"],
        params["lambda_0"],
        params["lambda_1"],
        params["mu"],
        params["alpha"],
        params["beta"]
    )

    scenario_Q_T = build_transient_generator(
        scenario_Q,
        params["K"]
    )

    for initial_index, regime in [
        (0, "Normal"),
        (params["K"], "Peak")
    ]:

        _, scenario_cdf, _ = (
            calculate_first_passage_distribution(
                scenario_Q_T,
                initial_index,
                comparison_times
            )
        )

        plt.plot(
            comparison_times,
            scenario_cdf,
            label=f"{scenario_name} - {regime}"
        )

plt.title("First-Passage CDF Under Different Parameter Scenarios")
plt.xlabel("Time")
plt.ylabel("Probability of Reaching Full Capacity")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "plot_12_sensitivity_cdf.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n==============================================")
print("ALL ADDITIONAL EXPERIMENTS COMPLETED")
print("==============================================")
