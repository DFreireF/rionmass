import matplotlib.pyplot as plt
from adjustText import adjust_text
import textalloc as ta

def iso_curve(revt, gammat, dp_p, sys, path = 108.36):
    return np.sqrt((((1-(path/(revt*(c/1e9)))**2-1/(gammat**2))*dp_p*revt)**2+sys**2))

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)

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



def isochronicity_curve_plot(T,T_y,T_n,sT, Nuclei_Y,Nuclei_N,ref,names_latex, gammat_0 = 1.395, 
                             labels = False, fast = True, fts = 36):
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))
    #indexes for proper plot
    index_with_ref = np.where(ref == 'Y')[0]
    index_without_ref = np.where(ref == 'N')[0]
    indexes_corrected = np.concatenate((index_with_ref,index_without_ref)).tolist()
    T = T[indexes_corrected]
    sT = sT[indexes_corrected]
    names_latex = names_latex[indexes_corrected]
    ##
    y_sigmaT_vals_Y = np.array([float(i) for i in np.array(Nuclei_Y)[:,8]])
    y_sigmaT_errs_Y = np.array([float(i) for i in np.array(Nuclei_Y)[:,9]])
    y_sigmaT_errs_N = np.array([])
    sigma = np.concatenate((y_sigmaT_errs_Y, y_sigmaT_errs_N))
    x_fit = np.arange((T/1000).min()*1e6,(T/1000).max()*1e6, 10000) / 1e6
    unknown_ions = int(len(T_n))
    if unknown_ions > 0:
        y_sigmaT_vals_N = np.array([float(i) for i in np.array(Nuclei_N)[:,8]])
        y_sigmaT_errs_N = np.array([float(i) for i in np.array(Nuclei_N)[:,9]])
        y_sigmaT_vals = np.concatenate((y_sigmaT_vals_Y, y_sigmaT_vals_N))
        y_sigmaT_errs = np.concatenate((y_sigmaT_errs_Y, y_sigmaT_errs_N))
    else:
        y_sigmaT_vals = y_sigmaT_vals_Y
        y_sigmaT_errs = y_sigmaT_errs_Y
        
    sigma = y_sigmaT_errs
    seeds = [gammat_0, 0.0008, 0.01]
    
    fit_params, fit_covariance = curve_fit(iso_curve, T/1000, sT, p0=seeds,
                                      sigma = sigma, absolute_sigma = True)
    Gammat,         dpop,       sigma_t_sys  = fit_params
    GammatError, dpopError, sigma_t_sysError = np.sqrt(np.diag(fit_covariance))
    chi2_fit_sigma = calculate_reduced_chi_squared(sT, iso_curve(T/1000, *fit_params), sigma,len(fit_params))
    sigma_sys = np.zeros_like(sigma)
    while chi2_fit_sigma > 1:
        sigma_sys += 0.00001
        fit_params, fit_covariance = curve_fit(iso_curve, T/1000, sT, p0=seeds,
                                      sigma = np.sqrt(sigma**2+sigma_sys**2), absolute_sigma = True)
        Gammat,         dpop,       sigma_t_sys  = fit_params
        GammatError, dpopError, sigma_t_sysError = np.sqrt(np.diag(fit_covariance))
        chi2_fit_sigma = calculate_reduced_chi_squared(sT, iso_curve(T/1000, *fit_params), np.sqrt(sigma**2+sigma_sys**2),len(fit_params))
    print(sigma_sys)  
    sigma_sys = sigma_sys[0]
    print('chi2_fit_sigma',chi2_fit_sigma)
    label_axs2 = f'$\sqrt{{\left(1 - \left(\\frac{{108.36}}{{T \cdot c}}\\right)^2 - \\frac{{1}}{{{Gammat:.4f}^2}}\\right)^2 \cdot \\left(\\frac{{{np.abs(dpop)*1e-1:.4f}}}{{100}} \cdot T\\right)^2 + {abs(sigma_t_sys):.4f}^2}}$'
    
    axs.errorbar(T_y, y_sigmaT_vals_Y, yerr=np.sqrt(y_sigmaT_errs_Y**2+sigma_sys**2), fmt='o', label="Calibrant ion",
                    markersize='13', ecolor='black',capsize=4, elinewidth=3, color='blue')
    if unknown_ions > 0:
        axs.errorbar(T_n, y_sigmaT_vals_N, yerr=np.sqrt(y_sigmaT_errs_N**2+sigma_sys**2), fmt='o', label="Unknown ion",
                        markersize='13', ecolor='black',capsize=4, elinewidth=3, color='orange')
    axs.plot(x_fit, iso_curve(x_fit, *fit_params), 'r-', label=label_axs2,  linewidth=5)
    axs.set_xlabel(r"Revolution time, $T$ (ns)", fontsize = fts)
    axs.set_ylabel(r"$\sigma_{T}$ (ps)", fontsize = fts)

    #axs.set_title(f"Iso-curve, $\\gamma_t = {Gammat:.4f}$, $\\frac{{\\sigma_p}}{{p}} = {np.abs(dpop)*1e-1:.4f}$ $\%$, $\\sigma_{{sys}} = {abs(sigma_t_sys):.4f}$ ps", fontsize = fts)
    axs.grid(True, linestyle=':', color='grey',alpha = 0.7)
    if labels:
        if fast:
            ta.allocate_text(fig,axs,T/1000,
                             sT,names_latex,x_scatter=T/1000, y_scatter=sT,
                textsize=35, nbr_candidates=5000, seed = 42, min_distance=0.05, max_distance=0.4)
        else:
            latex_labels = []
            for i,index in enumerate(indexes_corrected):
                latex_labels.append(axs.annotate(names_latex[i], (T[i]/1000, sT[i]), fontsize=36))
            adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='red'),alpha=.8)
            
    axs.legend(shadow=True,fancybox=True, fontsize = fts-5)
    axs.tick_params(axis='both', which='major', labelsize=fts)
    print(Gammat,         dpop,       sigma_t_sys, GammatError, dpopError, sigma_t_sysError)
    return axs

def moq_plot(T_y,T_n,Nuclei_Y,Nuclei_N,A_mins_y,A_mins_n):
    # Subplot 1: MoQ Plot
    
    
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))

    T_vals = np.concatenate((T_y,T_n))
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,15]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,14]]
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
        coefficient = float(A_mins[0][k])
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

def mass_excess_plot(ME_Exp_AME_y,ME_Exp_AME_n, MEError_Exp_y,MEError_Exp_n, 
                     MEError_AME_y,MEError_AME_n, MEError_sys_y,T_y,T_n, MEError_sys_n = [],fast = True, fts = 36):
    
    
    fig, axs = plt.subplots(1, 1, figsize=(25, 15))
    MEError_sys_n = [MEError_sys_y[0] for i in range(len(MEError_AME_n))]
    ME_Exp_AME_T = np.concatenate((ME_Exp_AME_y,ME_Exp_AME_n))
    exp_errors = np.concatenate((MEError_Exp_y,MEError_Exp_n))
    ame_errors = np.concatenate((MEError_AME_y,MEError_AME_n))
    systematic_errors = np.concatenate((MEError_sys_y,MEError_sys_n))
    statistical_errors = [np.sqrt(exp**2+ame**2) for exp,ame in zip(exp_errors,ame_errors)]
    
    indexx = len(MEError_Exp_n)
    # since I append the unknown errors at the end, I can access to them like this, and remove the AME error fromt them
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
    axs.set_xlabel(r"Revolution time, $T$ (ns)", fontsize = fts)
    axs.set_ylabel(r"$\Delta M_{\mathrm{exp}} - \Delta M_{\mathrm{AME}}$ (keV)", fontsize = fts)
    #axs.set_title("Mass excess difference")
    axs.grid(True, linestyle=':', color='grey',alpha = 0.7)
    
    
    index_with_ref = np.where(ref == 'Y')[0]
    index_without_ref = np.where(ref == 'N')[0]
    indexes_corrected = np.concatenate((index_with_ref,index_without_ref)).tolist()

    if fast:
        ta.allocate_text(fig,axs,T_T,ME_Exp_AME_T,names_latex[indexes_corrected],x_scatter=T_T, y_scatter=ME_Exp_AME_T,
                textsize=35, nbr_candidates=5000, seed = 42, min_distance=0.05, max_distance=0.4)
    else:
        latex_labels = []
        for i,index in enumerate(indexes_corrected):
            latex_labels.append(axs.annotate(names_latex[index], (T_T[i], ME_Exp_AME_T[i]), fontsize=36))
        adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='blue'),alpha=.8)
        
    axs.legend(shadow=True,fancybox=True, fontsize = fts-5)
    axs.tick_params(axis='both', which='major', labelsize=fts)
    return axs

def cmm_subplots(T_y,T_n,Nuclei_Y,Nuclei_N,A_mins_y,A_mins_n,T,sT,ref,names_latex,ME_Exp_AME_y,
                 ME_Exp_AME_n, MEError_Exp_y,MEError_Exp_n, MEError_AME_y,MEError_AME_n, 
                 MEError_sys_y,MEError_sys_n, gammat_0 = 1.395, fast = True):
    
    
    fig, axs = plt.subplots(2, 2, figsize=(25, 15))
    axs = axs.ravel()
    y_sigmaT_vals_Y = np.array([float(i) for i in np.array(Nuclei_Y)[:,8]])
    y_sigmaT_errs_Y = np.array([float(i) for i in np.array(Nuclei_Y)[:,9]])
    y_sigmaT_errs_N = np.array([])
    sigma = np.concatenate((y_sigmaT_errs_Y, y_sigmaT_errs_N))
    #moq subplot 1
    T_vals = np.concatenate((T_y,T_n))
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,15]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,14]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,15]]
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
        coefficient = float(A_mins[0][k])
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
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,15]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,14]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,15]]

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
                                      sigma = sigma, absolute_sigma = True)
    Gammat,         dpop,       sigma_t_sys  = fit_params
    GammatError, dpopError, sigma_t_sysError = np.sqrt(np.diag(fit_covariance))
    chi2_fit_sigma = calculate_reduced_chi_squared(sT, iso_curve(T/1000, *fit_params), sigma,len(fit_params))
    print('chi2_fit_sigma',chi2_fit_sigma)

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
    axs[3].errorbar(T_T, ME_Exp_AME_T, yerr=total_errors, fmt='bo', markersize=5, capsize=4, label='Total')
    axs[3].errorbar(T_T, ME_Exp_AME_T, yerr=statistical_errors, fmt='ro', markersize=5, capsize=4, label='Statistical')
    
    axs[3].axhline(y=0, color='red', linestyle='--')
    axs[3].set_xlabel(r"Revolution time, $T$ (ns)")
    axs[3].set_ylabel(r"$\Delta_{exp} - \Delta_{AME}$ (keV)")
    axs[3].set_title("Mass excess difference")
    axs[3].grid(True, linestyle=':', color='grey',alpha = 0.7)
    
    
    if fast:
        ta.allocate_text(fig,axs[3],T_T,ME_Exp_AME_T,names_latex[indexes_corrected],x_scatter=T_T, y_scatter=ME_Exp_AME_T,
                textsize=35, nbr_candidates=5000, seed = 42, min_distance=0.05, max_distance=0.4)
    else:
        latex_labels = []
        for i,index in enumerate(indexes_corrected):
            latex_labels.append(axs[3].annotate(names_latex[index], (T_T[i], ME_Exp_AME_T[i]), fontsize=36))
        adjust_text(latex_labels, arrowprops=dict(arrowstyle='->', color='blue'),alpha=.8)
    axs[3].legend(shadow=True,fancybox=True)
    plt.tight_layout()
    plt.show()

# Subplot 2: Residual Plot
def residual_plot(T_y,T_n,Nuclei_Y,Nuclei_N,A_mins_y,A_mins_n):

    fig, axs = plt.subplots(1, 1, figsize=(25, 15))
    
    moq_y = [float(i) for i in np.array(Nuclei_Y)[:,14]]
    moq_ye = [float(i) for i in np.array(Nuclei_Y)[:,15]]
    if len(Nuclei_N) > 0:
        moq_n =  [float(i) for i in np.array(Nuclei_N)[:,14]]
        moq_ne = [float(i) for i in np.array(Nuclei_N)[:,15]]

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

def plot_residual_distribution(by,bye, bins = 15):
    
    fig, axs = plt.subplots(1, 1, figsize=(25, 12))
    hist, bin_edges = np.histogram(by, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    initial_params = [110.0, 15.0, 12.0]  # Initial parameter guesses: amplitude, mean, stddev
    fit_params, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_params)
    x_bins_fit = np.arange(bin_centers.min()*10000,bin_centers.max()*10000,1000)/10000
    
    xerr_bars = []
    for i in range(len(bin_centers)):
        bin_mask = (by >= bin_edges[i]) & (by <= bin_edges[i + 1]) if i == len(bin_centers) - 1 else (by >= bin_edges[i]) & (by < bin_edges[i + 1])
        bin_y_error = bye[bin_mask]
        xerr_bars.append(np.mean(bin_y_error) if len(bin_y_error) > 0 else 0)
    
    width = (bin_edges[1] - bin_edges[0]) - 0.2
    # Shade areas for 1, 2, and 3 sigma
    sigma_values = [fit_params[2] * i for i in range(1, 4)]
    colors = ['blue', 'green', 'yellow']

    for sigma, color in zip(sigma_values, colors):
        rect = patches.Rectangle((fit_params[1] - sigma, 0), 2 * sigma, 1, linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
        axs.add_patch(rect)
    axs.bar(bin_centers,  hist/ hist.max(), width=width, color='r', xerr=xerr_bars,
           edgecolor='red', linewidth=2 )
    axs.errorbar(bin_centers,hist / hist.max(),xerr=xerr_bars,fmt='bo',ecolor='black',elinewidth=5,capsize=4,alpha=0.3,markersize='13')
    axs.plot(x_bins_fit, gaussian(x_bins_fit, *fit_params)/ hist.max(), 
             'b-', label=rf'$\mu$ = {np.round(fit_params[1])}, $\sigma$ = {np.round(fit_params[2])}', linewidth = 3)
    
    
        
    axs.set_title('Residuals distribution',  fontsize=38)
    axs.set_xlabel(r"$\Delta_{exp} - \Delta_{AME}$ (keV)", fontsize=34)
    axs.set_ylabel('Normalized amplitude', fontsize=34)
    axs.tick_params(axis='both', which='major', labelsize=34)
    axs.legend()
    plt.show()