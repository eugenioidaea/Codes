from sympy import symbols, integrate, exp, sqrt, pi

# Define variables
x, D, t, b = symbols('x D t b', positive=True)

# Define the integrand
integrand = (x * exp(-x**2 / (4 * D * t))) / sqrt(4 * pi * D * t**3)

# Perform the definite integral from 0 to b
result = integrate(integrand, (t, 0, b))
result.simplify()