import pandas as pd

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV et les transforme en DataFrame.

    Args:
    file_path : str : Le chemin du fichier CSV.

    Returns:
    pandas.DataFrame : Le DataFrame contenant les données du fichier CSV.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# Exemple d'utilisation
#file_path = 'data/raw/house_prices.csv'
#house_prices = load_data(file_path)
#print(house_prices.head()) 