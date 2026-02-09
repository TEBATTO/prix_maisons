
import pandas as pd
from prince import MCA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split

# Fonction split_data
def split_data(df, target, test_ratio=0.3, seed=13):
    """
    Sépare aléatoirement le jeu de données en ensembles d'entraînement et de test.
    
    Args:
    df : pandas.DataFrame : Le DataFrame contenant les données.
    target : str : Le nom de la variable à expliquer.
    test_ratio : float : La proportion de données à inclure dans l'ensemble de test (default=0.3).
    seed : int : La graine pour le générateur de nombres aléatoires (default=13).
    
    Returns:
    tuple : Les ensembles d'entraînement et de test pour les variables explicatives et la variable à expliquer.
    """
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
    return X_train, X_test, y_train, y_test


######### 

def encoder_var(df, variables):
    """
    Encode les variables catégorielles en remplaçant les modalités 'yes' et 'no' 
    par 'nom_variable_yes' et 'nom_variable_no'.

    Args:
    df : pandas.DataFrame : Le DataFrame contenant les variables à encoder.
    variables : list : Liste des variables catégorielles à encoder.

    Returns:
    pandas.DataFrame : Le DataFrame avec les variables encodées.
    """
    for var in variables:
            df[var] = df[var].replace({'yes': '1', 'no': '0'})

    return df

######## 

def acm(df, variables):
    """
    Réalise une Analyse des Correspondances Multiples (ACM) sur les variables catégorielles.

    Args:
    df : pandas.DataFrame : Le DataFrame contenant les variables à analyser.
    variables : list : Liste des variables catégorielles à analyser.

    Returns:
    prince.MCA : L'objet prince.MCA contenant les résultats de l'ACM.
    """
    
    
    # Réalisation de l'ACM
    mca = MCA(n_components=2, random_state=13) #mca = MCA(n_components=2, random_state=SEED)
    mca = mca.fit(df[variables])
    
    return mca



########

def plot_acm_modalities(mca, df, variables, highlight_variable='prix'):
    """
    Affiche la carte des modalités pour l'ACM.

    Args:
    mca : prince.MCA : L'objet prince.MCA contenant les résultats de l'ACM.
    df : pandas.DataFrame : Le DataFrame contenant les variables à analyser.
    variables : list : Liste des variables catégorielles à analyser.
    """
    
    
    # Récupération des coordonnées des modalités
    modalities = mca.column_coordinates(df[variables])
    
    # Affichage de la carte des modalités
    plt.figure(figsize=(10, 8))
    plt.scatter(modalities[0], modalities[1], color='red')
    
    # ajouter les axes x et y
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
    
    
    # Séparer les modalités de la variable mise en avant
    for i, modality in enumerate(modalities.index):
        if highlight_variable in modality:
            plt.scatter(modalities.iloc[i, 0], modalities.iloc[i, 1], color='blue', label=f"{highlight_variable} Modality" if i == 0 else None)
            plt.text(modalities.iloc[i, 0], modalities.iloc[i, 1], modality, fontsize=12, color='blue')
        else:
            plt.scatter(modalities.iloc[i, 0], modalities.iloc[i, 1], color='red')
            plt.text(modalities.iloc[i, 0], modalities.iloc[i, 1], modality, fontsize=10, color='black')
    
    plt.title('Carte des modalités - ACM')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True)
    plt.show()

    #######

# Transformation des variables catégorielles binaires en entier 0 et 1
# Fonction pour faire la transformation
def transformer_binaire_en_entier(df, variable):
    """
    Fonction pour transformer une variable binaire (avec 'oui'/'non' ou équivalent) 
    en entiers 0 et 1.
    
    Args:
    df : pandas.DataFrame : Le DataFrame contenant les variables à transformer.
    variable : str : Le nom de la variable à transformer.
    
    Returns:
    pandas.Series : La variable transformée en entiers 0 et 1.
    """
    # Remplacer 'yes' par '1' et 'no' par '0', puis convertir en int
    return df[variable].str.replace("yes", "1").str.replace("no", "0").astype(int)






