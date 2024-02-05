from ROOT import TCanvas, TFile, TProfile, TNtuple, TH1F, TH2F,TChain,TGraph,TDatime,TLine,TLatex,TPad,TLegend,TPaveText,TF1,TGraphErrors,TGaxis
from ROOT import gROOT, gBenchmark, gRandom, gSystem, gPad,gStyle
from io import StringIO
import numpy as np
import ctypes
import os
from array import *
import math
import sys
from prettytable import PrettyTable
from decimal import Decimal, getcontext
import mpmath
import sympy as sp
import ezodf
from barion.amedata import *
import matplotlib.pyplot as plt
from adjustText import adjust_text
from scipy.constants import *
from barion.particle import *
from barion.ring import *
from scipy.optimize import curve_fit

me          = physical_constants['electron mass energy equivalent in MeV'][0]
amu         = physical_constants['atomic mass constant energy equivalent in MeV'][0]

from loguru import logger



def LeastSquareFit(T, TError, MoQ, MoQError, p,iterationMax,A0,OutputMatrixOrNot,tol = None):
    N        =len(T)
    A        =A0[:]
    chi2_min =1e20
    A_min    =A[:]
    b2_inverse_min =[]
    chi2_previous = 0
    for iteration in range(0, iterationMax):
        FDB=[]
        PDB=[]
        # step 1: F matrix. This can go outside the loof
        for i in range(0, N):
            f   = T[i]
            row = []
            for k in range(0, p):
                row.insert(k,mpmath.power(f, k))
            FDB.append(row)
        # step 2 delta_MoQ_fi
        delta_MoQ_fiDB=[]
        for i in range(0, N):
            fi           = T[i]
            delta_fi     = TError[i]
            delta_MoQ_fi = 0
            for k in range(1, p):
                delta_MoQ_fi = delta_MoQ_fi + k*A[k]*mpmath.power(fi, k - 1)*delta_fi
            delta_MoQ_fiDB.insert(i,delta_MoQ_fi)
        #step 3 P matrix
        for i in range(0, N):
            row = []
            for j in range(0, N):
                a=0
                if i ==j:
                    a = 1/(float(MoQError[i])**2 + float(delta_MoQ_fiDB[i])**2)
                row.insert(j, a)
            PDB.insert(i,row)
        # step 4 chi2
        if iteration==0:
            chi2 = 0
            for i in range(0, N):
                fi        = T[i]
                delta_fi  = TError[i]
                y         = MoQ[i]
                ye        = MoQError[i]
                yfit      = 0
                yfit_error=delta_MoQ_fiDB[i]
                for k in range(0, p):
                    yfit = yfit + A[k]*mpmath.power(fi,k)
                chi2 = chi2 + (y - yfit)**2 / (ye**2  + yfit_error**2)
            chi2_min=chi2
            A_min   =A[:]
            if OutputMatrixOrNot: logger.info("iteration  = ",iteration," A_min = ",A_min," chi2_min=", chi2_min)
            
        # step 5 calculated parameters A
        F = mpmath.matrix(FDB)
        P = mpmath.matrix(PDB)
        F_T = F.transpose()
        a1 = P *  mpmath.matrix(MoQ)
        a2 = F_T * a1
        b1 = P * F
        b2 = F_T * b1
        b2_sympy = sp.Matrix(b2.tolist())
        det_b2 = b2_sympy.det()
        chi2 = 0        
        if det_b2 == 0:
            logger.info("b2 = FT*P*F is a singular matrix. Iteration stopped.")
            break
        else:
            b2_inverse = b2**-1
            A = b2_inverse * a2
            if iteration == 0: b2_inverse_min  = b2_inverse.copy()
            # step 6
            for i in range(0, N):
                fi        = T[i]
                delta_fi  = TError[i]
                y         = MoQ[i]
                ye        = MoQError[i]
                yfit      = 0
                yfit_error=delta_MoQ_fiDB[i]
                for k in range(0, p):
                    yfit = yfit + A[k]*mpmath.power(fi,k)
                chi2 = chi2 + (y - yfit)**2 / (ye**2  + yfit_error**2)
            if chi2_min>=chi2:
                chi2_min=chi2
                A_min   =A[:]
                b2_inverse_min  = b2_inverse.copy()
                if OutputMatrixOrNot: logger.info("iteration  = ",iteration," A_min = ",A_min," chi2_min=", chi2_min)
                #iteration = iteration
                
        delta_chi2 = abs(chi2 - chi2_previous)
        if tol:
            if delta_chi2 < tol:
                break
        chi2_previous = chi2
        if OutputMatrixOrNot: print_output(iteration, A, p, N, MoQ, MoQError, delta_MoQ_fiDB, T, TError, F, F_T, PDB, a1, a2, b1, b2, b2_inverse, chi2,A_min, chi2_min)
        logger.info("iteration  = ",iteration," A_min = ",A_min," chi2_min=", np.round(float(chi2_min)), "delta_chi2 =",np.round(float(delta_chi2))) 
    return A_min, chi2_min,b2_inverse_min
def print_output(iteration, A, p, N, MoQ, MoQError, delta_MoQ_fiDB, T, TError, F, F_T, PDB, a1, a2, b1, b2, b2_inverse, chi2, A_min, chi2_min):
    logger.info(f"Iteration {iteration} - Best Fit Parameters:")
    for k in range(p):
        logger.info(f"A[{k}] = {float(A_min[k]):.5e}")

    logger.info("\nData and Uncertainties:")
    logger.info("{:<15} {:<15} {:<15} {:<15}".format("MoQ", "MoQError", "delta_MoQ_fi", "T +/- TError"))
    for i in range(N):
        moq_val = float(MoQ[i])
        moq_err = float(MoQError[i])
        delta_moq_fi = float(delta_MoQ_fiDB[i])
        t_val = float(T[i])
        t_err = float(TError[i])

        logger.info("{:<15.5f} {:<15.5e} {:<15.5e} {:<15.5f} +/- {:<15.5f}".format(moq_val, moq_err, delta_moq_fi, t_val, t_err))

    logger.info("\nMatrices:")
    logger.info("F:")
    for i in range(N):
        logger.info("\t".join(f"{float(F[i, k]):.3f}" for k in range(p)))

    logger.info("F_T:")
    for i in range(p):
        logger.info("\t".join(f"{float(F_T[i, k]):.3e}" for k in range(N)))

    logger.info("P:")
    for i in range(N):
        for j in range(N):
            pdb_val = float(PDB[i][j])
            if i == j:
                logger.info("{:<6.3f}".format(pdb_val), end="\t")
            else:
                logger.info("{:<1}".format(pdb_val), end="\t")
        logger.info()

    logger.info("\nIntermediate Results:")
    logger.info("a1 = P * MoQ:")
    for i in range(N):
        logger.info(f"{float(a1[i]):.5f}")

    logger.info("a2 = F_T * P * MoQ:")
    for i in range(p):
        logger.info(f"{float(a2[i]):.5f}")

    logger.info("b1 = P * F:")
    for i in range(N):
        logger.info("\t".join(f"{float(b1[i, k]):.3f}" for k in range(p)))

    logger.info("b2 = F_T * P * F:")
    for i in range(p):
        logger.info("\t".join(f"{float(b2[i, k]):.3f}" for k in range(p)))

    logger.info("b2_inverse = (F_T * P * F)^-1:")
    for i in range(p):
        logger.info("\t".join(f"{float(b2_inverse[i, k]):.3e}" for k in range(p)))

    logger.info("\nFinal Chi-Squared:")
    logger.info(f"Chi-squared: {float(chi2):.5f} (Minimum: {float(chi2_min):.5f})")

def MassCalibration(Nuclei_Y,Nuclei_N,OutputMatrixOrNot_Y,p,A0_Y,iterationMax,tol = 0.1):
    Y_Y,YError_Y,X_Y,XError_Y = [[] for _ in range(4)]

    for i in range(0, len(Nuclei_Y)):
        Y_Y        .append(float(Nuclei_Y[i][13]))
        YError_Y   .append(float(Nuclei_Y[i][14]))
        X_Y        .append(float(Nuclei_Y[i][6]))
        XError_Y   .append(float(Nuclei_Y[i][7]))
    
    A_min_Y, chi2_min_Y,b2_inverse_min_Y = LeastSquareFit (X_Y, XError_Y, Y_Y, YError_Y, p,iterationMax,A0_Y, OutputMatrixOrNot_Y,tol)
    # Nuclei_N.insert(i_N,[element,Z,A,N,Q, Flag,T, Count, SigmaT, TError, ME, MEError, EBindingEnergy, Mass_Q, Mass_QError])
    
    table_list = []
    for i in range(0, len(Nuclei_N)):
        f        = float(Nuclei_N[i][6])
        delta_f  = float(Nuclei_N[i][7])
        MoQ_N    = 0
        for k in range(0, p):
            MoQ_N = MoQ_N +  A_min_Y[k]*mpmath.power(f,k)

        MoQ_N_fit_av=0
        for k in range(0, p):
            for l in range(0,p):
                MoQ_N_fit_av = MoQ_N_fit_av + b2_inverse_min_Y[k,l]*mpmath.power(f,k+l)
        MoQ_N_fit_av   = MoQ_N_fit_av**0.5

        n = len(Y_Y)
        MoQ_N_fit_sca  = (chi2_min_Y/(n-p))**0.5*MoQ_N_fit_av
        MoQ_N_fit_error= max(MoQ_N_fit_av, MoQ_N_fit_sca) # this formula is given in matos's thesis page 67.
        #MoQ_N_fit_error= MoQ_N_fit_av  # this formular is used the old fortran code.

        MoQ_N_fre_error= 0
        for k in range(1, p):
            MoQ_N_fre_error = MoQ_N_fre_error +  k*A_min_Y[k]*mpmath.power(f,k-1)*delta_f

        MoQ_N_tot_error = float((MoQ_N_fit_error**2 + MoQ_N_fre_error**2)**0.5)
        MoQ_N = float(MoQ_N)
        element          = Nuclei_N[i][0]
        Z_N              = int(Nuclei_N[i][1])
        A_N              = int(Nuclei_N[i][2])
        N_N              = int(Nuclei_N[i][3])
        Q_N              = float(Nuclei_N[i][4])
        Flag             = Nuclei_N[i][5]
        ME_N_AME         = float(Nuclei_N[i][10]) # MeV
        MEError_N_AME    = float(Nuclei_N[i][11]) # MeV
        EBindingEnergy_N = float(Nuclei_N[i][12]) # MeV
        MoQ_N_AME        = float(Nuclei_N[i][13]) # u/e
        MoQError_N_AME   = float(Nuclei_N[i][14]) # u/e
        Mass_N           = MoQ_N * Q_N * amu 
        ME_N             = Mass_N - A_N*amu + Q_N*me - EBindingEnergy_N#EBindingEnergy # MeV *********************
        MEError_N        = MoQ_N_tot_error/MoQ_N * Mass_N # MeV
        table_list.append([element, Flag,Z_N, A_N,Q_N, f/1000, (ME_N*1000-ME_N_AME*1000), (ME_N*1000), (MEError_N*1000), (MEError_N_AME*1000)])
    return table_list,A_min_Y,chi2_min_Y
def plot_residual_distribution(bx,by,bye, bins = 15):
    plt.style.use('/home/duskdawn/analysis/plt-style.mplstyle')
    plt.rc('axes', unicode_minus=False)
    
    fig, axs = plt.subplots(1, 1, figsize=(25, 12))
    bins = bins
    hist, bin_edges = np.histogram(by, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    initial_params = [110.0, 15.0, 12.0]  # Initial parameter guesses: amplitude, mean, stddev
    fit_params, err = curve_fit(gaussian, bin_centers, hist, p0=initial_params)
    x_bins_fit = np.arange(bin_centers.min()*10000,bin_centers.max()*10000,1000)/10000
    
    xerr_bars = []
    for i in range(len(bin_centers)):
        bin_mask = (by >= bin_edges[i]) & (by < bin_edges[i + 1])
        bin_y_error = bye[bin_mask]
        if len(bin_y_error) > 0:  # Check if the bin is not empty
            x_error_avg = np.mean(bin_y_error)
            xerr_bars.append(x_error_avg)
        else:
            xerr_bars.append(0.0)  # Assign a default value for empty bins
        
    width = (bin_edges[1] - bin_edges[0]) -0.2
    plt.bar(bin_centers,  hist/ hist.max(), width=width, color='r', xerr=xerr_bars,
           edgecolor='red', linewidth=2 )
    plt.errorbar(bin_centers,hist / hist.max(),xerr=xerr_bars,fmt='bo',ecolor='black',elinewidth=5,capsize=4,alpha=0.3,markersize='13')
    plt.plot(x_bins_fit, gaussian(x_bins_fit, *fit_params)/ hist.max(), 
             'b-', label=f'mean = {np.round(fit_params[1])}   std = {np.round(fit_params[2])}', linewidth = 3)
    plt.title('Residuals distribution',  fontsize=38)
    plt.xlabel(r"$\Delta_{exp} - \Delta_{AME}$ (keV)", fontsize=34)
    plt.ylabel('Normalized amplitud', fontsize=34)
    plt.xticks(fontsize=34)
    plt.yticks(fontsize=34)
    plt.legend()
    plt.show()
def convert_name(name):
    import re
    # Use regex to extract the atomic number, isotope name, and charge
    atomic_num, element, charge = re.findall(r"\d+|\D+", name)
    # Format the atomic number and charge as superscripts
    atomic_num = '$^{'+str(atomic_num)+'}$'
    charge = '$^{'+str(charge)+'+}$'

    # Form the element symbol by capitalizing the first letter of the isotope name
    element = element.capitalize()

    # Combine the superscripts and element symbol
    return atomic_num + element + charge
def strip_name(name):
    import re
    # Use regex to extract the atomic number, isotope name, and charge
    atomic_num, element, charge = re.findall(r"\d+|\D+", name)
    return atomic_num, element, charge
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
def initial_seeds_root(p,x_data,y_data):
    if p ==6:
        func_Title = "[0] + [1]*x*1e3 + [2]*x*x*1e6+ [3]*x*x*x*1e9+ [4]*x*x*x*x*1e12+[5]*x*x*x*x*x*1e15"
        func = TF1("func",func_Title,0,700)
        func.SetParameters(-3.05960e+00,1.20092e-05 ,-3.41612e-12,-1.41612e-18, -5.41612e-24, 5e-32)
    
    if p ==5:
        func_Title = "[0] + [1]*x*1e3 + [2]*x*x*1e6+ [3]*x*x*x*1e9+ [4]*x*x*x*x*1e12"
        func = TF1("func",func_Title,0,700)
        func.SetParameters(-3.05960e+00,1.20092e-05 ,-3.41612e-12,-1.41612e-18, -5.41612e-24)
    
    elif p==4:
        func_Title = "[0] + [1]*x*1e3 + [2]*x*x*1e6+ [3]*x*x*x*1e9"
        func = TF1("func",func_Title,0,700)
        func.SetParameters(-3.05960e+00,1.20092e-05 ,-3.41612e-12,-3.41612e-18)
    elif p==3:
        func_Title = "[0] + [1]*x*1e3 + [2]*x*x*1e6"
        func = TF1("func",func_Title,0,700)
        func.SetParameters(-3.05960e+00,1.20092e-05 ,-5.41612e-12)
    
    elif p==2:
        func_Title = "[0] + [1]*x*1e3"
        func = TF1("func",func_Title,0,700)
    
    elif p==1:
        func_Title = "[0]"
        func = TF1("func",func_Title,0,700)
        
    gT_MoQ_Y = TGraphErrors()
    gT_MoQ_Y.SetName("gT_MoQ_Y")
    gT_MoQ_Y.SetTitle("The dependance of moq on T for calibrant nuclei")
    for i_Y in len(x_data):
        gT_MoQ_Y.SetPoint(i_Y,x_data,y_data)
        gT_MoQ_Y.SetPointError(i_Y,TError,Mass_QError)
    func.Print()
    gT_MoQ_Y.Fit(func,"RN")
    A0_Y=[]
    for i in range(0,p):
        A0_Y.insert(i, func.GetParameter(i))
# Subplot 2: Residual Plot
def residual_plot(T_y,T_n,Nuclei_Y,Nuclei_N,A_mins_y,A_mins_n):
    plt.style.use('/home/duskdawn/analysis/plt-style.mplstyle')
    plt.rc('axes', unicode_minus=False)
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))
    
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,13]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,13]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,14]]

    y_residul_vals_Y, y_residul_errs_Y, y_residul_vals_N, y_residul_errs_N = [[] for _ in range(4)]
    for i in range(len(Nuclei_Y)):
        x = T_y[i]
        yfit = 0
        for k in range(0,p):
            yfit = yfit + A_mins_y[i][k]*(x*1000)**k
        y_residul_vals_Y.append(float(yfit - moq_y[i]))
        y_residul_errs_Y.append(float(moq_ye[i]))
    for i in range(len(Nuclei_N)):
        x = T_n[i]
        yfit = 0
        for k in range(0,p):
            yfit = yfit + A_mins_n[i][k]*(x*1000)**k    
        y_residul_vals_N.append(float(yfit - moq_n[i]))
        y_residul_errs_N.append(float(moq_ne[i]))
    
 
    axs.errorbar(T_y, np.array(y_residul_vals_Y)*1e6, yerr=np.array(y_residul_errs_Y)*1e6, fmt='o', label="Calibrant ion",
                    markersize='13', ecolor='black',capsize=4, elinewidth=3)
    if len(T_n) > 0:
        axs.errorbar(T_n, np.array(y_residul_vals_N)*1e6, yerr=np.array(y_residul_errs_N)*1e6, fmt='o', label="Unknown ion",
                        markersize='13', ecolor='black',capsize=4, elinewidth=3)
    axs.axhline(y=0, color='red', linestyle='--')
    axs.set_xlabel(r"Revolution time, $T$ (ns)")
    axs.set_ylabel(r"$\left(m/q\right)_{fit} - \left(m/q\right)_{AME}$ ($\mu$u/e)")
    axs.set_title("Fit Residuals")
    axs.grid(True, linestyle=':', color='grey',alpha = 0.7)
    axs.legend(shadow=True,fancybox=True)
    return axs
def iso_curve(revt, gammat, dp_p, sys, path = 108.36):
    return np.sqrt((((1-(path/(revt*(c/1e9)))**2-1/(gammat**2))*dp_p*revt)**2+sys**2))
def calculate_reduced_chi_squared(y_data, y_fit, yerror, num_params):
    # Calculate the residuals
    residuals = y_data - y_fit
    # Calculate the chi-squared
    chi_squared = np.sum(residuals**2/yerror**2)
    # Calculate the degrees of freedom (dof)
    dof = len(y_data) - num_params
    # Calculate the reduced chi-squared
    chi_squared_red = chi_squared / dof
    return chi_squared_red
def isochronicity_curve_plot(T,T_y,T_n,sT, Nuclei_Y,Nuclei_N,ref,names_latex, gammat_0 = 1.395, labels = False):
    plt.style.use('/home/duskdawn/analysis/plt-style.mplstyle')
    plt.rc('axes', unicode_minus=False)
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))
    
    y_sigmaT_vals_Y = np.array([float(i) for i in np.array(Nuclei_Y)[:,8]])
    y_sigmaT_errs_Y = np.array([float(i) for i in np.array(Nuclei_Y)[:,9]])
    y_sigmaT_errs_N = np.array([])
    x_fit = np.arange((T/1000).min()*1e6,(T/1000).max()*1e6, 10000) / 1e6
    axs.errorbar(T_y, y_sigmaT_vals_Y, yerr=y_sigmaT_errs_Y, fmt='o', label="Calibrant ion",
                    markersize='13', ecolor='black',capsize=4, elinewidth=3)
    if len(T_n) > 0:
        y_sigmaT_vals_N = [float(i) for i in np.array(Nuclei_N)[:,8]]
        y_sigmaT_errs_N = [float(i) for i in np.array(Nuclei_N)[:,9]]
        axs.errorbar(T_n, y_sigmaT_vals_N, yerr=y_sigmaT_errs_N, fmt='o', label="Unknown ion",
                        markersize='13', ecolor='black',capsize=4, elinewidth=3)
    label_axs2 = f'$\sqrt{{\left(1 - \left(\\frac{{108.36}}{{T \cdot c}}\\right)^2 - \\frac{{1}}{{\gamma_t^2}}\\right)^2 \cdot \\left(\\frac{{\sigma_{{p}}}}{{p}} \cdot T\\right)^2 + \sigma_{{sys}}^2}}$'
    
    sigma = np.concatenate((y_sigmaT_errs_Y, y_sigmaT_errs_N))
    seeds = [gammat_0, 0.0008, 0.01]
    fit_params, fit_covariance = curve_fit(iso_curve, T/1000, sT, p0=seeds,
                                      sigma = sigma, absolute_sigma = True)
    Gammat,         dpop,       sigma_t_sys  = fit_params
    GammatError, dpopError, sigma_t_sysError = np.sqrt(np.diag(fit_covariance))
    chi2_fit_sigma = calculate_reduced_chi_squared(sT, iso_curve(T/1000, *fit_params), sigma,len(fit_params))
    logger.info('chi2_fit_sigma',chi2_fit_sigma)
    axs.plot(x_fit, iso_curve(x_fit, *fit_params), 'r-', label=label_axs2,  linewidth=5)
    axs.set_xlabel(r"Revolution time, $T$ (ns)")
    axs.set_ylabel(r"$\sigma_{T}$ (ps)")
    
    axs.set_title(f"Iso-curve, $\\gamma_t = {Gammat:.4f}$, $\\frac{{\\sigma_p}}{{p}} = {np.abs(dpop)*1e-1:.4f}$ $\%$, $\\sigma_{{sys}} = {abs(sigma_t_sys):.4f}$ ps")
    
    axs.grid(True, linestyle=':', color='grey',alpha = 0.7)
    if labels:
        latex_labels = []
        index_with_ref = np.where(ref == 'Y')[0]
        index_without_ref = np.where(ref == 'N')[0]
        indexes_corrected = np.concatenate((index_with_ref,index_without_ref)).tolist()
        for i,index in enumerate(indexes_corrected):
            latex_labels.append(axs.annotate(names_latex[index], (T[i]/1000, sT[i]), fontsize=36))
        adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='red'),alpha=.8)
    axs.legend(shadow=True,fancybox=True)
    return axs
def moq_plot(T_y,T_n,Nuclei_Y,Nuclei_N,A_mins_y,A_mins_n):
    # Subplot 1: MoQ Plot
    plt.style.use('/home/duskdawn/analysis/plt-style.mplstyle')
    plt.rc('axes', unicode_minus=False)
    
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))

    T_vals = np.concatenate((T_y,T_n))
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,13]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,13]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,14]]
        A_mins = np.concatenate((A_mins_y,A_mins_n))
    else: 
        moq_n = []
        moq_ne = []
        A_mins = A_mins_y
    
    moq_vals = np.concatenate((moq_y,moq_n))
    moq_errs = np.concatenate((moq_ye,moq_ne))
    moq_fit = np.zeros_like(T_vals)
    for i,x in enumerate(T_vals):
        yfit = 0
        for k in range(p):
            yfit += A_mins[i][k] * (x * 1000) ** k
        moq_fit[i] = yfit
        
    axs.errorbar(T_vals, moq_vals, yerr=moq_errs, fmt='o', 
                    markersize='10', ecolor='black',capsize=4, elinewidth=3, label = f'AME 2020')
    
    text_params = []
    for k in range(0,p):
        coefficient = float(A_min_Y[k])
        exponent = int(mpmath.log10(abs(coefficient)))
        mantissa = coefficient / 10 ** exponent
        param = f"$a_{{{k}}} =$"
        if coefficient > 0: param = param + f' $+$ '
        param = param + f"${mantissa:.1f} \\cdot 10^{{{exponent}}}$"
        text_params.append(param)
    text_params_box_info = "\n".join(text_params)
    
    pol_fit_formula = []
    for k in range(0,p):
        term = f"$a_{{{k}}} \\cdot T^{{{k}}}$"
        pol_fit_formula.append(term)
    fit_label_pol = "+".join(pol_fit_formula)
    
    axs.plot(T_vals, moq_fit, 'r-', label=fit_label_pol, linewidth = 3)
    axs.set_xlabel(r"Revolution time, $T$ (ns)")
    axs.set_ylabel(r"$\left(m/q\right)_{AME20}$ (u/e)")
    axs.set_title("Mass-to-charge")
    axs.grid(True, linestyle=':', color='grey',alpha = 0.7)
    axs.legend(shadow=True,fancybox=True)
    axs.text(0.7, 0.2, text_params_box_info, transform=axs.transAxes, fontsize=25, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    return axs
def mass_excess_plot(ME_Exp_AME_y,ME_Exp_AME_n, MEError_Exp_y,MEError_Exp_n, MEError_AME_y,MEError_AME_n, MEError_sys_y,MEError_sys_n,T_y,T_n):
    plt.style.use('/home/duskdawn/analysis/plt-style.mplstyle')
    plt.rc('axes', unicode_minus=False)
    
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))
    
    ME_Exp_AME_T = np.concatenate((ME_Exp_AME_y,ME_Exp_AME_n))
    exp_errors = np.concatenate((MEError_Exp_y,MEError_Exp_n))
    ame_errors = np.concatenate((MEError_AME_y,MEError_AME_n))
    systematic_errors = np.concatenate((MEError_sys_y,MEError_sys_n))
    statistical_errors = [np.sqrt(exp**2+ame**2) for exp,ame in zip(exp_errors,ame_errors)]
    indexx = len(MEError_Exp_n)
    if indexx > 0:
        statistical_errors[-indexx:] = exp_errors[-indexx:]
        
    T_T = np.concatenate((T_y,T_n))
    total_errors = [np.sqrt(l**2+m**2) for l,m in zip(statistical_errors,systematic_errors)]
    systematic_errors_plot = [np.sqrt(total_errors**2-statistical_errors**2)+statistical_errors for total_errors,statistical_errors in zip(total_errors,statistical_errors)]

    # Set different colors for the error bars
    #axs.errorbar(T_T, ME_Exp_AME_T, yerr=systematic_errors_plot, fmt='bo', markersize=5, capsize=4, label='Systematic')
    axs.errorbar(T_T, ME_Exp_AME_T, yerr=total_errors, fmt='bo', markersize=5, capsize=4, label='Total')
    axs.errorbar(T_T, ME_Exp_AME_T, yerr=statistical_errors, fmt='ro', markersize=5, capsize=4, label='Statistical')
    
    axs.axhline(y=0, color='red', linestyle='--')
    axs.set_xlabel(r"Revolution time, $T$ (ns)")
    axs.set_ylabel(r"$\Delta_{exp} - \Delta_{AME}$ (keV)")
    axs.set_title("Mass excess difference")
    axs.grid(True, linestyle=':', color='grey',alpha = 0.7)
    
    
    index_with_ref = np.where(ref == 'Y')[0]
    index_without_ref = np.where(ref == 'N')[0]
    indexes_corrected = np.concatenate((index_with_ref,index_without_ref)).tolist()
    latex_labels = []
    for i,index in enumerate(indexes_corrected):
        latex_labels.append(axs.annotate(names_latex[index], (T_T[i], ME_Exp_AME_T[i]), fontsize=36))
    adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='blue'),alpha=.8)
    axs.legend(shadow=True,fancybox=True)
    return axs
def cmm_subplots(T_y,T_n,Nuclei_Y,Nuclei_N,A_mins_y,A_mins_n,T,sT,ref,names_latex,ME_Exp_AME_y,
                 ME_Exp_AME_n, MEError_Exp_y,MEError_Exp_n, MEError_AME_y,MEError_AME_n, 
                 MEError_sys_y,MEError_sys_n, gammat_0 = 1.395):
    plt.style.use('/home/duskdawn/analysis/plt-style.mplstyle')
    plt.rc('axes', unicode_minus=False)
    
    fig, axs = plt.subplots(2, 2, figsize=(25, 15))
    axs = axs.ravel()

    #moq subplot 1
    T_vals = np.concatenate((T_y,T_n))
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,13]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,13]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,14]]
        A_mins = np.concatenate((A_mins_y,A_mins_n))
    else: 
        moq_n = []
        moq_ne = []
        A_mins = A_mins_y
    
    moq_vals = np.concatenate((moq_y,moq_n))
    moq_errs = np.concatenate((moq_ye,moq_ne))
    moq_fit = np.zeros_like(T_vals)
    for i,x in enumerate(T_vals):
        yfit = 0
        for k in range(p):
            yfit += A_mins[i][k] * (x * 1000) ** k
        moq_fit[i] = yfit
        
    axs[0].errorbar(T_vals, moq_vals, yerr=moq_errs, fmt='o', 
                    markersize='10', ecolor='black',capsize=4, elinewidth=3, label = f'AME 2020')
    
    text_params = []
    for k in range(0,p):
        coefficient = float(A_min_Y[k])
        exponent = int(mpmath.log10(abs(coefficient)))
        mantissa = coefficient / 10 ** exponent
        param = f"$a_{{{k}}} =$"
        if coefficient > 0: param = param + f' $+$ '
        param = param + f"${mantissa:.1f} \\cdot 10^{{{exponent}}}$"
        text_params.append(param)
    text_params_box_info = "\n".join(text_params)
    
    pol_fit_formula = []
    for k in range(0,p):
        term = f"$a_{{{k}}} \\cdot T^{{{k}}}$"
        pol_fit_formula.append(term)
    fit_label_pol = "+".join(pol_fit_formula)
    
    axs[0].plot(T_vals, moq_fit, 'r-', label=fit_label_pol, linewidth = 3)
    axs[0].set_xlabel(r"Revolution time, $T$ (ns)")
    axs[0].set_ylabel(r"$\left(m/q\right)_{AME20}$ (u/e)")
    axs[0].set_title("Mass-to-charge")
    axs[0].grid(True, linestyle=':', color='grey',alpha = 0.7)
    axs[0].legend(shadow=True,fancybox=True)
    axs[0].text(0.7, 0.2, text_params_box_info, transform=axs[0].transAxes, fontsize=25, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    #Residual subplot 2
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,13]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,13]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,14]]

    y_residul_vals_Y, y_residul_errs_Y, y_residul_vals_N, y_residul_errs_N = [[] for _ in range(4)]
    for i in range(len(Nuclei_Y)):
        x = T_y[i]
        yfit = 0
        for k in range(0,p):
            yfit = yfit + A_mins_y[i][k]*(x*1000)**k
        y_residul_vals_Y.append(float(yfit - moq_y[i]))
        y_residul_errs_Y.append(float(moq_ye[i]))
    for i in range(len(Nuclei_N)):
        x = T_n[i]
        yfit = 0
        for k in range(0,p):
            yfit = yfit + A_mins_n[i][k]*(x*1000)**k    
        y_residul_vals_N.append(float(yfit - moq_n[i]))
        y_residul_errs_N.append(float(moq_ne[i]))
    axs[1].errorbar(T_y, np.array(y_residul_vals_Y)*1e6, yerr=np.array(y_residul_errs_Y)*1e6, fmt='o', label="Calibrant ion",
                    markersize='13', ecolor='black',capsize=4, elinewidth=3)
    if len(T_n) > 0:
        axs[1].errorbar(T_n, np.array(y_residul_vals_N)*1e6, yerr=np.array(y_residul_errs_N)*1e6, fmt='o', label="Unknown ion",
                        markersize='13', ecolor='black',capsize=4, elinewidth=3)
    axs[1].axhline(y=0, color='red', linestyle='--')
    axs[1].set_xlabel(r"Revolution time, $T$ (ns)")
    axs[1].set_ylabel(r"$\left(m/q\right)_{fit} - \left(m/q\right)_{AME}$ ($\mu$u/e)")
    axs[1].set_title("Fit Residuals")
    axs[1].grid(True, linestyle=':', color='grey',alpha = 0.7)
    axs[1].legend(shadow=True,fancybox=True)
    
    #iso curve subplot 3
    fit_params, fit_covariance = curve_fit(iso_curve, T/1000, sT, p0=[gammat_0, 0.0001, 0.001],
                                      sigma = eT, absolute_sigma = True)
    Gammat,         dpop,       sigma_t_sys  = fit_params
    GammatError, dpopError, sigma_t_sysError = np.sqrt(np.diag(fit_covariance))
    chi2_fit_sigma = calculate_reduced_chi_squared(sT, iso_curve(T/1000, *fit_params), eT,len(fit_params))
    logger.info('chi2_fit_sigma',chi2_fit_sigma)

    y_sigmaT_vals_Y = [float(i) for i in np.array(Nuclei_Y)[:,8]]
    y_sigmaT_errs_Y = [float(i) for i in np.array(Nuclei_Y)[:,9]]
    
    
    x_fit = np.arange((T/1000).min()*1e6,(T/1000).max()*1e6, 10000) / 1e6
    axs[2].errorbar(T_y, y_sigmaT_vals_Y, yerr=y_sigmaT_errs_Y, fmt='o', label="Calibrant ion",
                    markersize='13', ecolor='black',capsize=4, elinewidth=3)
    if len(T_n) > 0:
        y_sigmaT_vals_N = [float(i) for i in np.array(Nuclei_N)[:,8]]
        y_sigmaT_errs_N = [float(i) for i in np.array(Nuclei_N)[:,9]]
        axs[2].errorbar(T_n, y_sigmaT_vals_N, yerr=y_sigmaT_errs_N, fmt='o', label="Unknown ion",
                        markersize='13', ecolor='black',capsize=4, elinewidth=3)
    label_axs = f'$\sqrt{{\left(1 - \left(\\frac{{108.36}}{{T \cdot c}}\\right)^2 - \\frac{{1}}{{\gamma_t^2}}\\right)^2 \cdot \\left(\\frac{{\sigma_{{p}}}}{{p}} \cdot T\\right)^2 + \sigma_{{sys}}^2}}$'
    
    axs[2].plot(x_fit, iso_curve(x_fit, *fit_params), 'r-', label=label_axs,  linewidth=5)
    axs[2].set_xlabel(r"Revolution time, $T$ (ns)")
    axs[2].set_ylabel(r"$\sigma_{T}$ (ps)")
    
    axs[2].set_title(f"Iso-curve, $\\gamma_t = {Gammat:.4f}$, $\\frac{{\\sigma_p}}{{p}} = {dpop*1e-1:.4f}$ $\%$, $\\sigma_{{sys}} = {abs(sigma_t_sys):.4f}$ ps")
    
    axs[2].grid(True, linestyle=':', color='grey',alpha = 0.7)
    #latex_labels = []
    index_with_ref = np.where(ref == 'Y')[0]
    index_without_ref = np.where(ref == 'N')[0]
    indexes_corrected = np.concatenate((index_with_ref,index_without_ref)).tolist()
    #for i,index in enumerate(indexes_corrected):
    #    latex_labels.append(axs[2].annotate(names_latex[index], (T[i]/1000, sT[i]), fontsize=36))
    #adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='red'),alpha=.8)
    axs[2].legend(shadow=True,fancybox=True)
    
    #mass excess subplot 4
    ME_Exp_AME_T = np.concatenate((ME_Exp_AME_y,ME_Exp_AME_n))
    exp_errors = np.concatenate((MEError_Exp_y,MEError_Exp_n))
    ame_errors = np.concatenate((MEError_AME_y,MEError_AME_n))
    systematic_errors = np.concatenate((MEError_sys_y,MEError_sys_n))
    statistical_errors = [np.sqrt(exp**2+ame**2) for exp,ame in zip(exp_errors,ame_errors)]
    indexx = len(np.where(ref=='N'))
    if indexx > 0:
        statistical_errors[-indexx:] = exp_errors[-indexx:]
    T_T = np.concatenate((T_y,T_n))
    total_errors = [np.sqrt(l**2+m**2) for l,m in zip(statistical_errors,systematic_errors)]
    # Set different colors for the error bars
    axs[3].errorbar(T_T, ME_Exp_AME_T, yerr=total_errors, fmt='bo', markersize=5, capsize=4, label='Systematic')
    axs[3].errorbar(T_T, ME_Exp_AME_T, yerr=statistical_errors, fmt='ro', markersize=5, capsize=4, label='Statistical')
    
    axs[3].axhline(y=0, color='red', linestyle='--')
    axs[3].set_xlabel(r"Revolution time, $T$ (ns)")
    axs[3].set_ylabel(r"$\Delta_{exp} - \Delta_{AME}$ (keV)")
    axs[3].set_title("Mass excess difference")
    axs[3].grid(True, linestyle=':', color='grey',alpha = 0.7)
    
    
    #index_with_ref = np.where(ref == 'Y')[0]
    #index_without_ref = np.where(ref == 'N')[0]
    #indexes_corrected = np.concatenate((index_with_ref,index_without_ref)).tolist()
    latex_labels = []
    for i,index in enumerate(indexes_corrected):
        latex_labels.append(axs[3].annotate(names_latex[index], (T_T[i], ME_Exp_AME_T[i]), fontsize=36))
    adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='blue'),alpha=.8)
    axs[3].legend(shadow=True,fancybox=True)
    plt.tight_layout()
    plt.show()