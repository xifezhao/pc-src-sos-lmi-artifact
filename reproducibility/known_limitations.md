# Known Limitations

1. Exact SDP wall-clock solve times (`ms`) depend on CPU clock speed and BLAS threading.
2. Linearized `H_inf <= 1` certificates guarantee small-signal disturbance attenuation around cruising equilibrium; severe emergency braking engages nonlinear IDM and actuator saturation bounds, which are evaluated separately via nonlinear ODE Monte Carlo and adversarial grid search.
