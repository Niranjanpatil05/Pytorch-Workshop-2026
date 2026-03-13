def dydx_simple(x):
    """Calcualte derivative of y = x^2 manually"""
    return 2 * x 
x_val = 2
derivative = dydx_simple(x_val)
print(f"dy/dx at x={x_val} is {derivative}")