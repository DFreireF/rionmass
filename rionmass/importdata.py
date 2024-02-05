import requests
from bs4 import BeautifulSoup

def Read_AME(file_path = "AME.rd"):
    try:
        # Try to open the file and read its contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        # If the file is not found, print an error message
        logger.info(f"Error: File '{file_path}' not found.")
        return
    logger.info(f"Reading mass data from file '{file_path}'...")
    # Initialize an empty list to store parsed data
    # Iterate through each import requests
from bs4 import BeautifulSoupline and parse the data
    data_list= []
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if line:  # Skip empty lines
            tokens = line.split()  # Split the line by spaces
            if len(tokens) == 6:  # Ensure each line has 6 fields
                entry = {
                    'Element': tokens[0],
                    'Z': int(tokens[1]),
                    'A': int(tokens[2]),
                    'ME (keV)': float(tokens[3]),
                    'ME_error (keV)': float(tokens[4]),
                    'Symbol': tokens[5] #this is not necessary
                }
                data_list.append(entry)
    return data_list
def Read_AME_barion():
    ame = AMEData()  
    data_list= []
    for i in ame.ame_table:
        entry = {
            'Element': str(i[5])+i[6],
            'Z': int(i[4]),
            'A': int(i[5]),
            'ME (keV)': float(i[8]),
            'ME_error (keV)': float(i[9])
                }
        data_list.append(entry)
    return data_list
def Read_elbien_file(file_path):
    data_list = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        logger.info(f"Error: File '{file_path}' not found.")
        return data_list
    logger.info(f"Reading electron binding energy data from file '{file_path}'...")
    header_line = None
    for line in lines:
        if line.startswith('#'):
            if header_line is None:
                header_line = line
            continue
        
        if not line.strip():
            continue  # Skip empty lines
        
        tokens = line.split()
        if header_line:
            headers = header_line.strip().split()[1:]  # Extract column headers
            entry = {'Z': int(tokens[0])}
            
            entry['TotBEn'] = int(tokens[1])
            # Add more fields up to '105e-ion'
            for i in range(len(headers), len(tokens)-2):
                entry[f'{i + 1}e-ion'] = int(tokens[i+2])
            data_list.append(entry)
    return data_list
def Read_revlutiontime(file_path = "revlutiontime.txt"):
    try:
        # Try to open the file and read its contents
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        # If the file is not found, print an error message
        logger.info(f"Error: File '{file_path}' not found.")
        return
    logger.info(f"Reading revolution time from file '{file_path}'...")
    # Initialize an empty list to store parsed data
    # Iterate through each line and parse the data
    data_list = []
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if line.startswith('#'):
            continue
        if line:  # Skip empty lines
            tokens = line.split()  # Split the line by spaces
            entry = {
                'Element': tokens[0],
                'Z': int(float(tokens[1])),
                'Q': int(float(tokens[2])),
                'Flag': tokens[3],
                'RevolutionTime[ns]': float(tokens[4]),
                'Count': float(tokens[5]),
                'width of revolutionTime[ps]': float(tokens[6]),
                'error of width of revolutionTime[ps]': float(tokens[7])
            }
            data_list.append(entry)
    
    # Print the parsed data
    logger.info("Total nuclei number: ",len(data_list))
    for entry in range (0, len(data_list)):
        logger.info(data_list[entry])
    logger.info("...")
        
    return data_list

def GetAMEData(AME_data,element):    
    for entry_AME in AME_data:     
        if element == entry_AME['Element']:
            ME,MEError,A,N= entry_AME['ME (keV)']/1e3, entry_AME['ME_error (keV)']/1e3, entry_AME['A'], entry_AME['A'] - entry_AME['Z']
    return ME, MEError, A, N

def GetBindingEnergy(elbien_data,Z,Q):
    match = next((entry for entry in elbien_data if entry['Z'] == Z), None)
    if match:
        key = 'TotBEn' if Z - Q == 0 else f"{Q}e-ion"
        return match.get(key, 0) / 1e6
    return 0

def _retrieve_electron_binding_energies():
    # URL of the website
    url = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
    
    # Parameters for the POST request
    data = {
        'spectra': 'H-Og',
        'submit': 'Retrieve Data',
        'units': '1',
        'format': '0',
        'order': '0',
        'at_num_out': 'on',
        'ion_charge_out': 'on',
        'e_out': '1',
        'unc_out': 'on'
    }
    
    # Send a POST request to the website
    response = requests.post(url, data=data)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table in the HTML
        table = soup.find('table', {'border': '1'})
        
        if table:
            # Initialize the dictionary
            data_dict = {}
    
            # Iterate over the rows of the table, skipping the header row
            for row in table.find_all('tr')[1:]:
                cells = row.find_all('td')
                if len(cells) == 4:
                    at_num = cells[0].text.strip()
                    ion_charge = cells[1].text.strip()
                    binding_energy = cells[2].text.strip()
                    uncertainty = cells[3].text.strip()
    
                    # Create a key-value pair in the dictionary
                    if at_num not in data_dict:
                        data_dict[at_num] = []
                    data_dict[at_num].append({
                        'Ion Charge': ion_charge,
                        'Binding Energy (eV)': binding_energy,
                        'Uncertainty (eV)': uncertainty
                    })
    
            # Print the dictionary
            for key, values in data_dict.items():
                print(f"At. Num. {key}:")
                for value in values:
                    print(f"  {value}")
        else: print("Table not found in the HTML content.")
    else: print("Failed to retrieve data from the website.")
    return data_dict

def _save_electron_binding_energies(data_dict, filename = 'electron_binding_energies'):
    npz_data = {key: np.array(value) for key, value in data_dict.items()}
    # Save the data to a .npz file
    np.savez(filename, **npz_data)

def _load_electron_binding_energies(filename = 'electron_binding_energies.npz'):
    data = np.load(filename, allow_pickle=True)
    # Convert the loaded data back to a dictionary format
    return {key: data[key].tolist() for key in data}

def _convert_binding_energy_to_float(s):
    # Remove square brackets and parentheses
    s = s.strip('[]()')
    s = s.replace(' ', '')
    return float(s)

def _get_ZQ_binding_energy_and_uncertainty(data_dict,Z,Q):
    ion = data_dict[f'{Z}'][Z-Q]
    binding_energy = _convert_binding_energy_to_float(ion['Binding Energy (eV)'])
    uncertainty = float(ion['Uncertainty (eV)'])
    return binding_energy, uncertainty

def _get_electron_binding_energies(filename = 'electron_binding_energies.npz'):
    if os.path.exists(filename):
        return _load_electron_binding_energies(filename)
    else: 
        data_dict = _retrieve_electron_binding_energies()
        _save_electron_binding_energies(data_dict)
        return data_dict
    