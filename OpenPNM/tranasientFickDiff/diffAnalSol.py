from sympy import symbols, integrate, exp, sqrt, pi

# Define variables
x, D, t, t1 = symbols('x D t t1', positive=True)

# Define the integrand
# integrand = (x * exp(-x**2 / (4 * D * t))) / sqrt(4 * pi * D * t**3)
integrand = exp(-x**2 / (4 * D * t)) / sqrt(4 * pi * D * t)
# integrand = (x * exp(-x**2 * (t - D)**2 / (4 * D * t))) / sqrt(4 * pi * D * t**3)

# Perform the definite integral from 0 to b
result = integrate(integrand, (t, 0, t1))
result.simplify()