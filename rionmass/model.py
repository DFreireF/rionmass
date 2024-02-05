import numpy as np
import mpmath
import sympy as sp

def LeastSquareFit(T, TError, MoQ, MoQError, p,iterationMax,A0,OutputMatrixOrNot,tol = None):
    N        =len(T)
    A        = A0.copy()
    chi2_min =1e20
    A_min    =A.copy()
    b2_inverse_min =[]
    chi2_previous = 1e20
    chi2 = 0 
    FDB = [[mpmath.power(f, k) for k in range(p)] for f in T]
    for iteration in range(iterationMax):
        delta_MoQ_fiDB= compute_delta_MoQ_fi(T, TError, A, p)
        PDB = [[1 / (MoQError[i] ** 2 + delta_MoQ_fiDB[i] ** 2) if i == j else 0 for j in range(N)] for i in range(N)]
        F = mpmath.matrix(FDB)
        P = mpmath.matrix(PDB)
        F_T = F.transpose()
        a1 = P *  mpmath.matrix(MoQ)
        a2 = F_T * a1
        b1 = P * F
        b2 = F_T * b1
        b2_sympy = sp.Matrix(b2.tolist())
        det_b2 = b2_sympy.det()
        
        if det_b2 == 0:
            raise ValueError("b2 = FT*P*F is a singular matrix. Iteration stopped.")
            
        b2_inverse = b2**-1
        A = b2_inverse * a2
        chi2 = compute_chi_squared(N, T, TError, MoQ, MoQError, A, p, delta_MoQ_fiDB)

        if chi2<chi2_min:
            chi2_min=chi2
            A_min   =A.copy() 
            if OutputMatrixOrNot: print("iteration  = ",iteration," A_min = ",A_min," chi2_min=", chi2_min)
                
        delta_chi2 = abs(chi2 - chi2_previous)
        if tol and delta_chi2 < tol:
                break
        chi2_previous = chi2
        
        if OutputMatrixOrNot: print_output(iteration, A, p, N, MoQ, MoQError, delta_MoQ_fiDB, T, TError, F, F_T, PDB, a1, a2, b1, b2, b2_inverse, chi2,A_min, chi2_min)
        print(f"Iteration = {iteration}, A_min = {A_min}, chi2_min = {np.round(float(chi2_min))}, delta_chi2 = {np.round(float(delta_chi2))}")
    return A_min, chi2_min,b2_inverse

def compute_delta_MoQ_fi(T, TError, A, p):
    delta_MoQ_fiDB = []
    for i, (fi, delta_fi) in enumerate(zip(T, TError)):
        delta_MoQ_fi = sum(k * A[k] * mpmath.power(fi, k - 1) * delta_fi for k in range(1, p))
        delta_MoQ_fiDB.append(delta_MoQ_fi)
    return delta_MoQ_fiDB

def compute_chi_squared(N, T, TError, MoQ, MoQError, A, p, delta_MoQ_fiDB):
    chi2 = 0
    for i in range(N):
        yfit = sum(A[k] * mpmath.power(T[i], k) for k in range(p))
        chi2 += (MoQ[i] - yfit) ** 2 / (MoQError[i] ** 2 + delta_MoQ_fiDB[i] ** 2)
    return chi2

def MassCalibration(Nuclei_Y,Nuclei_N,OutputMatrixOrNot_Y,p,A0_Y,iterationMax,tol = 0.1):
    Y_Y = [float(nucleus[14]) for nucleus in Nuclei_Y]
    YError_Y = [float(nucleus[15]) for nucleus in Nuclei_Y]
    X_Y = [float(nucleus[6]) for nucleus in Nuclei_Y]
    XError_Y = [float(nucleus[7]) for nucleus in Nuclei_Y]
    
    #LeastSquareFit
    A_min_Y, chi2_min_Y,b2_inverse_min_Y = LeastSquareFit (X_Y, XError_Y, Y_Y, YError_Y, p,iterationMax,A0_Y, OutputMatrixOrNot_Y,tol)
    
    table_list = []
    for nucleus in Nuclei_N:
        f = float(nucleus[6])
        delta_f = float(nucleus[7])

        # Calculate MoQ_N 
        MoQ_N = sum(A_min_Y[k] * mpmath.power(f, k) for k in range(p))

        # Calculate MoQ_N_fit_av 
        MoQ_N_fit_av = sum(b2_inverse_min_Y[k, l] * mpmath.power(f, k + l) for k in range(p) for l in range(p))
        MoQ_N_fit_av = mpmath.sqrt(MoQ_N_fit_av)

        MoQ_N_fit_sca = mpmath.sqrt(chi2_min_Y / (len(Y_Y) - p)) * MoQ_N_fit_av
        MoQ_N_fit_error = max(MoQ_N_fit_av, MoQ_N_fit_sca)  # 67th page of Matos's thesis
        #MoQ_N_fit_error= MoQ_N_fit_av  # this formular is used the old fortran code.
        MoQ_N_fre_error = sum(k * A_min_Y[k] * mpmath.power(f, k - 1) * delta_f for k in range(1, p))
        MoQ_N_tot_error = float(mpmath.sqrt(MoQ_N_fit_error**2 + MoQ_N_fre_error**2))   
        
        MoQ_N = float(MoQ_N)
        element          = nucleus[0]
        Z_N              = int(nucleus[1])
        A_N              = int(nucleus[2])
        N_N              = int(nucleus[3])
        Q_N              = float(nucleus[4])
        Flag             = nucleus[5]
        ME_N_AME         = float(nucleus[11]) # MeV
        MEError_N_AME    = float(nucleus[12]) # MeV
        EBindingEnergy_N = float(nucleus[13]) # MeV
        MoQ_N_AME        = float(nucleus[14]) # u/e
        MoQError_N_AME   = float(nucleus[15]) # u/e
        Mass_N           = MoQ_N * Q_N * amu 
        ME_N             = Mass_N - A_N*amu + Q_N*me - EBindingEnergy_N#EBindingEnergy # MeV *********************
        MEError_N        = MoQ_N_tot_error/MoQ_N * Mass_N # MeV
        table_list.append([element, Flag,Z_N, A_N,Q_N, f/1000, (ME_N*1000-ME_N_AME*1000), (ME_N*1000), (MEError_N*1000), (MEError_N_AME*1000)])
    return table_list,A_min_Y,chi2_min_Y

def self_calibration(Nuclei_Y, p, A0_Y, iterationMax):
    print("# 1. self-calibration.")
    # Goes over every reference ion and compute their m/q from the fit considering all others known but it
    table = PrettyTable()
    table.field_names = ["Ion", "Ref?", "TOF (ns)", "ME(EXP-AME)(keV)", "ME_EXP(keV)", 
                         "ERROR(EXP,keV)", "ERROR(AME,keV)"]
    
    # Initialize lists to store data
    chi_min_list, T_y, ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y, A_mins_y,table_listDB = [], [], [], [], [], [], []
    
    # Iterate over each nucleus for self-calibration
    for i, nucleus_y in enumerate(Nuclei_Y):
        Nuclei_Y_SelfCal = [Nuclei_Y[j] for j in range(len(Nuclei_Y)) if i != j]
        Nuclei_N_SelfCal = [nucleus_y]
    
        table_list, A_min_Y, chi2_min_Y = MassCalibration(Nuclei_Y_SelfCal, Nuclei_N_SelfCal, False, p, A0_Y, iterationMax, tol=1e-10)
    
        # Extract and store calibration results
        A_mins_y.append(A_min_Y)
        calibration_result = table_list[0]
        table_listDB.append(calibration_result)
        
        # Adding row to the table
        ion_label = f"{calibration_result[0]}{int(calibration_result[4])}+"
        table.add_row([ion_label, calibration_result[1]] + ["%10.6f" % value for value in calibration_result[5:10]])
        
        # Append results to lists
        T_y.append(calibration_result[5])
        ME_Exp_AME_y.append(calibration_result[6])
        MEError_Exp_y.append(calibration_result[8])
        MEError_AME_y.append(calibration_result[9])
        chi_min_list.append(chi2_min_Y / (len(Nuclei_Y) - p))
        
        # Print statement for each nucleus
        print(f"unknown nucleus is {ion_label}, ME(exp-ame) = {calibration_result[6]:.3f} \u00B1 {calibration_result[8]:.3f} keV")
    
    # Print the entire table
    print(table)
    return chi_min_list, T_y, ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y, A_mins_y,table_listDB,table
def determination_of_systematic_error(table_listDB, ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y, Nuclei_Y, p):
    
    MEError_Exp_array = np.array([entry[9] for entry in table_listDB])
    chi2_a = sum((me_exp_ame**2 / (me_error_exp**2 + me_error_ame**2)) 
                 for me_exp_ame, me_error_exp, me_error_ame in zip(ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y))
    
    # Initial normalized chi-squared calculation
    chin_a = chi2_a / (len(Nuclei_Y) - 1 - p) # 1 for the one we consider as unknown in the self calibration
    
    # Systematic error adjustment loop
    MEError_sys, chin_b = 0,0
    if chin_a > 1:
        while True:
            chi2_b = sum((me_exp_ame**2 / (me_error_exp**2 + me_error_ame**2 + MEError_sys**2)) 
                         for me_exp_ame, me_error_exp, me_error_ame in zip(ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y))
            chin_b = chi2_b / (len(table_listDB) - p -1)
    
            if chin_b <= 1:
                break
    
            MEError_sys += 0.01
    
    MEError_sys_y = [MEError_sys] * len(table_listDB)
    
    info_table = PrettyTable()

    # Define the table columns
    info_table.field_names = ["Description", "Value"]
    
    # Add rows to the table
    info_table.add_row(["Fitting function", f"m/q(T)=a0+a1*T+a2*T^2+...+an*T^n"])
    info_table.add_row(["Fit order n", p - 1])
    #info_table.add_row(["A_min_Y", A_mins_y])
    info_table.add_row(["Normalized χ²", chin_a])
    info_table.add_row(["Normalized χ² (+systematic error)", chin_b])
    info_table.add_row(["Systematic error", f"{MEError_sys} keV"])
    info_table.add_row(["REF_NUC NO", len(Nuclei_Y)])
    
    # Print the table
    print(info_table)
    return MEError_sys_y

def calibration_of_unknown_nuclei(Nuclei_Y, Nuclei_N, p, A0_Y, iterationMax):
    print("# 2. mass calibration for nuclei with unknown mass.")
    table_u = PrettyTable()
    table_u.field_names = ["Ion", "Ref?", "TOF (ns)", "ME(EXP-AME)(keV)", "ME_EXP(keV)", 
                           "ERROR(EXP,keV)", "ERROR(AME,keV)"]
    
    if len(Nuclei_N) > 0:
        table_list, A_min_Y, chi2_min_Y = MassCalibration(Nuclei_Y, Nuclei_N, False, p, A0_Y, iterationMax, tol=1e-10)
        MEError_Exp_array = np.array([])
        T_n, ME_Exp_AME_n, MEError_Exp_n, MEError_AME_n = [], [], [], []
    
        for entry in table_list:
            ion_label = f"{entry[0]}{int(entry[4])}+"
            table_u.add_row([ion_label, entry[1]] + ["%10.6f" % value for value in entry[5:10]])
            MEError_Exp_array = np.append(MEError_Exp_array, entry[8])
            T_n.append(entry[5])
            ME_Exp_AME_n.append(entry[6])
            MEError_Exp_n.append(entry[8])
            MEError_AME_n.append(entry[9])
        
        print("Fitting function: m/q(T)=a0+a1*T+a1*T^2+...+a1*T^n")
        print("Fit order n:", p - 1)
        print("A_min_Y:", A_min_Y)
        print("normalized χ²:", chi2_min_Y / (len(Nuclei_Y) - p))
        print("REF_NUC NO:", len(Nuclei_Y))
    
        # Print the table
        print(table_u)
        return T_n, ME_Exp_AME_n, MEError_Exp_n, MEError_AME_n
    else:
        print("No nuclei with unknown mass for calibration.")
        return [], [], [], []
