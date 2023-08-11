order = 2
scaling = 1

# Creates the initial weights (seeds)
weights = np.array([1 / (scaling**2 * freqe[i]**2) for i in range(len(atomic_mass_u))])


# Iterative process until convergence is obtained (masses does not change anymore)
for iteration in range(0,10000):
    #if iteration % 10000 == 0: print(iteration)
    if iteration > 0:
        # We update the weights based on the new a_vect obtained (a_vect are the parameters of the polynomial we are fitting)
        weights = [1 / np.sum([(k*a_vect[k]*freq**(k-1)*freqe[i]) for k in range(1,order+1)])**2 for i,_ in enumerate(atomic_mass_u)]

    # Generate A matrix
    A = np.zeros((order+1)*(order+1)).reshape(order+1,order+1)
    for i in range(0,order+1):
        for j in range(0,order+1):
            if iteration == 0: A[i,j]= np.sum([weights[k]*freq[k]**(i+j) for k,_ in enumerate(atomic_mass_u)])
            else: A[i,j] = np.sum([weights[k]*freq[k]**(i+j) for k,_ in enumerate(atomic_mass_u)])
    # Compute and inverse matrix A            
    Ainv = np.linalg.inv(A)            
    f_vect = np.array([[freq[i]**k for k in range(0,order+1)] for i in range(len(atomic_mass_u))])
    # Find the new parameters of the fit by multipling A inversed with the sum
    if iteration == 0: 
        a_vect = Ainv @ sum([weights[i] * ionic_mass_u[i]/int(charge[i]) * f_vect[i] for i in range(len(atomic_mass_u))])
    else: 
        a_vect = Ainv @ sum([weights[i] * (cmass[i] + benergy_u[i]) / int(charge[i]) * f_vect[i] for i in range(len(atomic_mass_u))])
    # We create the W matrix, in order to compute the masses
    W = np.zeros(len(atomic_mass_u)*len(atomic_mass_u)).reshape(len(atomic_mass_u),len(atomic_mass_u))

    for i in range(len(atomic_mass_u)):
        for j in range(len(atomic_mass_u)):
            W[i,j] = - weights[i] / charge[i] * weights[j] / charge[j] * f_vect[i] @ Ainv @ f_vect[j]
            if j == i: 
                W[i,j] = W[i,j] + weights[j] / charge[j]**2
                if int(references[j]) == 1: 
                    W[i,j] = W[i,j] + 1 / atomic_mass_error_u[j]**2
    Winv = np.linalg.inv(W)
    # We inverser the matrix, and create the v vector
    v = np.zeros(len(atomic_mass_u))
    for j in range(len(atomic_mass_u)):
        beta = np.sum([weights[i] / charge[i] * f_vect[i] @ Ainv @ f_vect[j] * benergy_u[i] for i in range(len(atomic_mass_u))])
        B = weights[j] / charge[j] * beta - weights[j] * benergy_u[j] / charge[j] / charge[j]
        v[j] = B
        if references[j] == 1: 
            v[j] = v[j] + atomic_mass_u[j] / atomic_mass_error_u[j]**2
    # We calculate the new masses obtained in this iteration with the found parameters of the polynomial
    cmass = Winv @ v
    # We feed again the new masses into the iteration in order to minimize the likelihood function
