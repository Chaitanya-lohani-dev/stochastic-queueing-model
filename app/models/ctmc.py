import numpy as np

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


k =int(input("Enter the maximum number of customers in the system (k): "))
lambda_0 = float(input("Enter the arrival rate in state 0 (lambda_0): "))
lambda_1 = float(input("Enter the arrival rate in state 1 (lambda_1): "))
mu = float(input("Enter the service rate (mu): "))
alpha = float(input("Enter the transition rate from state 0 to state 1 (alpha): "))
beta = float(input("Enter the transition rate from state 1 to state 0 (beta): "))

Q = build_mmpp_generator(k, lambda_0, lambda_1, mu, alpha, beta)
print("Q \n", Q)

validation_result = validate_generator(Q, k)
print(validation_result)