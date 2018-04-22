from sympy import *

init_session(use_latex=True)
init_printing(use_latex=True)
x = symbols('x')
Integral(sqrt(1 / x), x)

n = 3
u1, u2, u3 = symbols('u1 u2 u3')
u_mean = (u1 + u2 + u3) / n
way1 = (u1 - u_mean) ** 2 + (u2 - u_mean) ** 2 + (u3 - u_mean) ** 2
way2 = (u1 - u2) ** 2 + (u1 - u3) ** 2 + (u2 - u3) ** 2
simplify(way1)
simplify(way2)

expand(way1)
expand(way2)
simplify(way1 / way2)

z = symbols('z')
n = symbols('n')
summation(n * z ** -n, (n, -oo, +oo))
summation((1 + n) / 3 ** n * z ** -n, (n, 0, +oo))

simplify(1/ ( 3*z * ( 1- 1/z/3)**2 ) + 1/ (1- 1/(3*z)))
