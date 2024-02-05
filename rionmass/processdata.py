import ezodf
import mpmath


def process_sheet(sheet, exclusion_list):
    data = []
    # careful with the format of the ods. It must have 7 columns even if the 2nd is irrelevant, modify
    for row in range(sheet.nrows()):
        if row in exclusion_list:
            continue

        row_data = [sheet[row, col].value for col in range(sheet.ncols()) if sheet[row, col].value is not None]
        if row_data:
            # Ensure row_data has a consistent length
            row_data += [None] * (sheet.ncols() - len(row_data))
            data.append(row_data)

    # Convert to a 2D numpy array with dtype=object to handle mixed data types
    return np.array(data, dtype=object)

def get_processed_data(sheet_data, unknown = None):
    names = sheet_data[:, 0]
    harmonics = sheet_data[:, 2].astype(int)
    f = sheet_data[:, 3].astype(float)
    fe = sheet_data[:, 4].astype(float)
    s = sheet_data[:, 5].astype(float)
    se = sheet_data[:, 6].astype(float)

    names_latex = [convert_name(name) for name in names]
    stripped = np.array([strip_name(name) for name in names])
    atomic_num, element, charge = stripped[:,0], stripped[:,1], stripped[:,2]
    name_aux = [str(a) + str(e) for a, e in zip(atomic_num, element)]

    T = 1e12 / (f / harmonics)
    sT = s / f * T
    eT =   harmonics / f / f / f * (se*f+2*s*fe) *1e12
    er = fe / (f**2) * harmonics * 1e12
    
    zz = []
    for ele in element:
        for i in AMEData().ame_table:
            if str(i[6]) == str(ele):
                zz.append(int(i[4]))
                break
    
    references = ['Y' for _ in charge]
    if unknown:
        for unknown_ion in unknown:
            index = np.where(names == unknown_ion)[0][0]
            references[index] = 'N'
    data = np.column_stack([name_aux, zz, charge, references, T, er, sT, eT,names_latex])
    return data

def process_nuclei(data, AME_data, elbien_data):
    '''
    Nucleus lists with
    0       1 2 3 4    5  6     7      8          9         10     11     12          13           14      15
    element,Z,A,N,Q, Flag,T, Terror, SigmaT, SigmaTError, latex,   ME, MEError, EBindingEnergy, Mass_Q, Mass_QError
    '''
    
    Nuclei, Nuclei_Y, Nuclei_N = [], [], []

    for ion in data:
        Z, Q = int(ion[1]), int(ion[2])  # Extract Z and Q once and reuse
        ME, MEError, A, N = GetAMEData(AME_data, ion[0])  # MeV
        EBindingEnergy, EBindingEnergy_uncertainty = _get_ZQ_binding_energy_and_uncertainty(elbien_data, Z, Q)
        
        # Convert to MeV and calculate Mass and Mass_Q
        EBindingEnergy /= 10**6  # MeV
        EBindingEnergy_uncertainty /= 10**6  # MeV
        Mass = A * amu + ME - Q * me + EBindingEnergy
        Mass_error = np.sqrt(MEError**2 + EBindingEnergy_uncertainty**2)
        Mass_Q = Mass / (amu * Q)
        Mass_QError = Mass_error / (amu * Q)

        # Build the complete nucleus data list
        nucleus = ion.tolist() + [ME, MEError, EBindingEnergy, Mass_Q, Mass_QError]
        nucleus.insert(2, A)
        nucleus.insert(3, N)
        Nuclei.append(nucleus)
        
        # Append to Nuclei_Y or Nuclei_N based on Flag
        if ion[3] == "Y": Nuclei_Y.append(nucleus)
        if ion[3] == "N": Nuclei_N.append(nucleus)

    # Reporting
    if Nuclei_N:
        print(f"Total nuclei number: {len(Nuclei)}, calibrant nuclei: {len(Nuclei_Y)}, unknown nuclei: {len(Nuclei_N)}, First unknown nucleus: {Nuclei_N[0][0]}")
    else:
        print(f"Total nuclei number: {len(Nuclei)}, calibrant nuclei: {len(Nuclei_Y)}, unknown nuclei: {len(Nuclei_N)}, None")
    
    return Nuclei, Nuclei_Y, Nuclei_N

def get_initial_seeds(p, nuclei_data):

    # Ensure nuclei_data is a NumPy array
    nuclei_data = np.asarray(nuclei_data)

    # Extracting T, Mass_Q, and Mass_QError from nuclei_data
    T = nuclei_data[:, 6].astype(float) / 1000
    Mass_Q = nuclei_data[:, 14].astype(float)
    Mass_QError = nuclei_data[:, 15].astype(float)

    # Get initial parameter estimates and covariance matrix
    params, cov = initial_seeds(p, T, Mass_Q, Mass_QError)
    print(f'intiial fit errors = {np.sqrt(np.diag(cov))}')
    
    return params

def initial_seeds(p,x_data,y_data,y_data_err):
    if p ==7:
        def pol7(x,a,b,c,d,e,f,g):
            return a + b*x*1e3 + c*x**2*1e6+ d*x**3*1e9+ e*x**4*1e12 + f*x**5*1e15+g*x**6*1e18
        initial_guess = [-3.05960e+00,1.20092e-05 ,-3.41612e-12,-1.41612e-18, -5.41612e-24, 5e-32,-5e-37]
        params, covariance = curve_fit(pol7, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    if p ==6:
        def pol6(x,a,b,c,d,e,f):
            return a + b*x*1e3 + c*x**2*1e6+ d*x**3*1e9+ e*x**4*1e12 + f*x**5*1e15
        initial_guess = [-3.05960e+00,1.20092e-05 ,-3.41612e-12,-1.41612e-18, -5.41612e-24, 5e-32]
        params, covariance = curve_fit(pol6, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    if p ==5:
        def pol5(x,a,b,c,d,e):
            return a + b*x*1e3 + c*x**2*1e6+ d*x**3*1e9+ e*x**4*1e12
        initial_guess = [-3.05960e+00,1.20092e-05 ,-3.41612e-12,-1.41612e-18, -5.41612e-24]
        params, covariance = curve_fit(pol5, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    elif p==4:
        def pol4(x,a,b,c,d):
            return a + b*x*1e3 + c*x**2*1e6+ d*x**3*1e9
        initial_guess = [-3.05960e+00,1.20092e-05 ,-3.41612e-12,-3.41612e-18]
        params, covariance = curve_fit(pol4, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    elif p==3:
        def pol3(x,a,b,c):
            return a + b*x*1e3 + c*x**2*1e6
        initial_guess = [-3.05960e+00,1.20092e-05 ,-3.41612e-12]
        params, covariance = curve_fit(pol3, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    elif p==2:
        def pol2(x,a,b):
            return a + b*x*1e3
        initial_guess = [0,1]
        params, covariance = curve_fit(pol2, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    elif p==1:
        def pol1(x,a):
            return a
        initial_guess = [0]
        params, covariance = curve_fit(pol1, x_data, y_data, p0=initial_guess, sigma = y_data_err, absolute_sigma = True)
    return params, covariance

def define_precision(p):
    if p ==3:
        precision = 20
    elif p ==4:
        precision = 31
    elif p==5:
        precision = 42
    elif p ==3:
        precision = 18
    elif p == 6:
        precision = 54
    elif p == 7:
        precision = 65
    else:
        precision = None
    return precision