import numpy as np

def InexactLineSearchMethod(f_func, f_deriv, alpha_init, direction, x_init=0, f_bar=-np.inf, rho=1e-2, sigma = 0.1, tau=9, bracketing_max_iterations=50,
                            tau2=0.1, tau3=0.5,sectioning_max_iterations=10):
    f0 = f_func(x_init)
    fd0 = f_deriv(x_init)
    f_prime0 = np.dot(fd0, direction)

    if f_prime0 >= 0:
        raise ValueError("No descending :c")

    if f_bar > -np.inf:
        mu = (f_bar - f0) / (rho * f_prime0)
    else:
        mu = np.inf
    
    alpha_prev = 0.0
    alpha_curr = min(alpha_init, mu) if mu > 0 else alpha_init
    bracket_found = False

    a = 0.0
    b = 0.0

    print("== Bracketing phase ==")
    print(f"{'it':>3} | {'alpha':>8} | {'f(alpha)':>12} | {'f_der(alpha)':>12}")

    for i in range(bracketing_max_iterations):
        
        #Evaluate at current alpha
        x_curr = x_init + alpha_curr * direction
        phi_curr = f_func(x_curr)

        if phi_curr <= f_bar:
            return alpha_curr
        
        armijo_bound = f0 + alpha_curr*rho*f_prime0
        grad_curr = f_deriv(x_curr)
        phi_prime_curr = np.dot(grad_curr, direction)
        phi_prev = f_func(x_init + alpha_prev*direction)

        print(f"{i} | {alpha_curr} | {phi_curr} | {phi_prime_curr}")

        if (phi_curr > armijo_bound) or (phi_curr >= phi_prev):
            a = alpha_prev
            b = alpha_curr
            bracket_found = True
            break

        if abs(phi_prime_curr) <= - sigma * f_prime0:
            return alpha_curr
        
        if phi_prime_curr >= 0:
            a = alpha_curr
            b = alpha_prev
            bracket_found = True
            break

        if mu <= 2*alpha_curr - alpha_prev:
            alpha_next = mu
        else:
            lower_bound = 2 * alpha_curr - alpha_prev
            upper_bound = min(mu, alpha_curr + tau*(alpha_curr - alpha_prev))
            alpha_next = 0.5 * (lower_bound + upper_bound)

        alpha_prev = alpha_curr
        alpha_curr = alpha_next

    if not bracket_found:
        return alpha_curr
    
    if a > b:
        a, b = b, a
    
    print("\n=== Sectioning phase ===")
    print(f"Initial bracket: [{a}, {b}]")

    f_a = f_func(x_init + a * direction)

    for j in range(sectioning_max_iterations):
        lower_bound = a + tau2 * (b - a)
        upper_bound = b - tau3 * (b - a)
        
        # Not completely sure about this step, because you can choose wichever 
        # value between both bounds
        x_curr = 0.5 * (lower_bound + upper_bound)

        x_point = x_init + x_curr * direction
        armijo_bound_x_curr = f0 + rho * x_curr * f_prime0
        f_x_point = f_func(x_point)
        grad_currj = f_deriv(x_point)
        f_prime_currj = np.dot(grad_currj, direction)

        print(f"{j} | {x_curr} | {f_x_point} | {f_prime_currj} | bracket= [{a}, {b}]")

        if (f_x_point > armijo_bound_x_curr) or (f_x_point >= f_a):
            a_next = a
            b_next = x_curr
        else:
            if abs(f_prime_currj) <= -sigma*f_prime0:
                return x_curr

            a_next = x_curr
            f_a = f_x_point

            b_next = a if f_prime_currj * (b - a) >= 0 else b

        a, b = a_next, b_next

    alpha_curr = 0.5 * (a + b)
    return alpha_curr