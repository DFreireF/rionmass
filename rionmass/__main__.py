import argparse
import ezodf
#import the other libraries

def controller(filename, revname, sheet_index, exclusion_list, p, iterationMax, unknown = None):
    # Importing data
    odsinfo = ezodf.opendoc(filename)
    sheet = odsinfo.sheets[sheet_index]
    sheet_data = process_sheet(sheet, exclusion_list)
    processed_data = get_processed_data(sheet_data, unknown = unknown)
    
    save_data_to_file(revname, processed_data)
    precision = define_precision(p)
    mpmath.mp.dps = precision
    # Read external info
    AME_data            = Read_AME_barion()
    elbien_data = _get_electron_binding_energies()
    # AME values for each nuclei
    Nuclei, Nuclei_Y, Nuclei_N = process_nuclei(processed_data, AME_data, elbien_data)
    
    A0_Y = get_initial_seeds(p, Nuclei_Y)
    #self calibration
    chi_min_list, T_y, ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y, A_mins_y,table_listDB,table = self_calibration(Nuclei_Y, p, A0_Y, iterationMax)
    MEError_sys_y = determination_of_systematic_error(table_listDB, ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y, Nuclei_Y, p)
    finaltable = final_table(table, MEError_sys_y[0])
    #mass calibration unknown nuclei
    T_n, ME_Exp_AME_n, MEError_Exp_n, MEError_AME_n = calibration_of_unknown_nuclei(Nuclei_Y, Nuclei_N, p, A0_Y, iterationMax)
    
    return processed_data,T_n, ME_Exp_AME_n, MEError_Exp_n, MEError_AME_n, MEError_sys_y, chi_min_list, T_y, ME_Exp_AME_y, MEError_Exp_y, MEError_AME_y, A_mins_y,table_listDB,table, Nuclei, Nuclei_Y, Nuclei_N

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the mass analysis.')
    parser.add_argument('filename', type=str, help='Input file name (ODS format).')
    parser.add_argument('revname', type=str, help='Revolution times file name for output.')#This should be removed or optional
    parser.add_argument('sheet_index', type=int, help='Sheet index in the ODS file.')
    parser.add_argument('exclusion_list', nargs='+', type=int, help='List of row indices to exclude.')
    parser.add_argument('p', type=int, help='Precision for calculations.')
    parser.add_argument('iterationMax', type=int, help='Maximum number of iterations for calibration.')
    parser.add_argument('--unknown', type=str, default=None, help='Optional unknown ion for processing.')

    args = parser.parse_args()

    results = controller(args.filename, args.revname, args.sheet_index, args.exclusion_list, args.p, args.iterationMax, args.unknown)