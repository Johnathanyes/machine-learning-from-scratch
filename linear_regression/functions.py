import copy
import numpy as np
import math

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

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        fw_b = w * x[i] + b
        dj_dw_i = (fw_b - y[i]) * x[i]
        dj_db_i = (fw_b - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient
      
    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b] 
      """
    
    w = copy.deepcopy(w_in) # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w , b)     

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            
 
    return w, b