import numpy as np

def compute_cost(x: np.array, y: np.array, w: int, b: int):
    """
    Computes the cost function for linear regression.
    
    arguments:
      x (ndarray (m,)): data, m examples 
      y (ndarray (m,)): target values
      w (scalar): rate of change for linear regression 
      b (scalar): y itercept for linear regression
    
    returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    num_examples = x.shape[0]
    cost_sum = 0
    for i in range(num_examples):
        fw_b = w * x[i] + b
        instance_cost = (fw_b - y[i]) ** 2
        cost_sum += instance_cost
    total_cost = cost_sum / (2 * num_examples)
    return total_cost


def compute_gradient(x: np.array, y: np.array, w: int, b: int):
    return 0