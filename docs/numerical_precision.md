# Numerical Precision and Conditioning

Polynomial variables `(x, y, z, c, sigma)` are dimensionlessly scaled onto `[-1, 1]` and normalized by characteristic amplitude `S_y = max |D_nom|(x)` prior to SDP assembly. Every SDP solution is audited post-solve for minimum symmetric Gram eigenvalue `>= -5e-6` and maximum monomial identity residual `<= 5e-5`.
