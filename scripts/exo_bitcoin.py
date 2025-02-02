import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Chargement des données avec gestion d'erreur en cas de fichier introuvable
file_path = r"C:\Users\enzoc\Desktop\Data_Analysis_Python\data_set\BTC-EUR.csv"

try:
    bitcoin = pd.read_csv(file_path, parse_dates=True, index_col='Date')  # Lecture du CSV avec la colonne "Date" en index
    print("Fichier chargé avec succès !")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable.")
    exit()

# Affichage des informations de base sur le dataset
print("\nDimensions du dataset :", bitcoin.shape)
print("Colonnes disponibles :", bitcoin.columns)
print("\nAperçu des premières lignes :")
print(bitcoin.head())

# Vérification de la présence de la colonne "Close"
if 'Close' in bitcoin.columns:
    plt.figure(figsize=(12, 6))
    bitcoin['Close'].plot(title="Évolution du prix du Bitcoin en EUR", color="blue", label="Prix quotidien")
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture (EUR)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()
else:
    print("\nErreur : La colonne 'Close' n'est pas présente dans le dataset. Vérifiez le fichier CSV.")
    exit()

# --- FILTRAGE ET RÉSAMPLAGE DES DONNÉES ---

plt.figure(figsize=(14, 7))

# Courbe des prix quotidiens en 2019
bitcoin.loc['2019', 'Close'].plot(label="Prix quotidien", color='black', alpha=0.7)

# Moyenne mensuelle
bitcoin.loc['2019', 'Close'].resample('M').mean().plot(label="Moyenne mensuelle", linestyle='dashed', color='red')

# Moyenne hebdomadaire
bitcoin.loc['2019', 'Close'].resample('W').mean().plot(label="Moyenne hebdomadaire", linestyle='dashed', color='green')

# Personnalisation du graphique
plt.title("Analyse du prix du Bitcoin en 2019 avec différentes échelles temporelles")
plt.xlabel("Date")
plt.ylabel("Prix de clôture (EUR)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# Calcul des statistiques par semaine (moyenne, écart-type, min, max)
m = bitcoin.loc['2019', 'Close'].resample('W').agg(['mean', 'std', 'min', 'max'])

# Affichage des statistiques dans la console
print("\nStatistiques hebdomadaires du Bitcoin en 2019 :")
print(m)

# Création du graphique
plt.figure(figsize=(12, 6))

# Tracé de la moyenne hebdomadaire
plt.plot(m.index, m['mean'], label="Moyenne hebdomadaire", color="blue")

# Remplissage de l'intervalle min-max par semaine
plt.fill_between(m.index, m['max'], m['min'], color='gray', alpha=0.3, label='Min-Max par semaine')

# Personnalisation du graphique
plt.title("Prix du Bitcoin en 2019 : Moyenne et Intervalle Min-Max Hebdomadaire")
plt.xlabel("Date")
plt.ylabel("Prix de clôture (EUR)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

# Affichage du graphique
plt.show()

bitcoin.loc['2019', 'Close'].rolling(window=7).mean().plot()
plt.show()

# --- FILTRAGE ET RÉSAMPLAGE DES DONNÉES ---

plt.figure(figsize=(14, 7))

# Courbe des prix quotidiens en 2019
bitcoin.loc['2019-09', 'Close'].plot(label="Prix quotidien", color='black', alpha=0.7)

# Moyenne mobile sur 7 jours non centrée
bitcoin.loc['2019-09', 'Close'].rolling(window=7).mean().plot(label="Moyenne mobile 7 jours non-centrée", linestyle=':', color='blue')

# Moyenne mobile sur 7 jours centrée
bitcoin.loc['2019-09', 'Close'].rolling(window=7, center=True).mean().plot(label="Moyenne mobile 7 jours centrée", linestyle=':', color='green')

# Moyenne mobile sur 7 jours ewm
bitcoin.loc['2019-09', 'Close'].ewm(alpha = 0.6).mean().plot(label="Moyenne mobile 7 jours ewm", linestyle=':', color='red')


# Personnalisation du graphique
plt.title("Analyse du prix du Bitcoin en 2019 avec différentes échelles temporelles")
plt.xlabel("Date")
plt.ylabel("Prix de clôture (EUR)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Affichage des prix du Bitcoin en septembre 2019 avec différents facteurs d'exponentielle
plt.figure()

# Affichage des prix réels
bitcoin.loc['2019-09', 'Close'].plot(label="Prix réel")

# Ajout des courbes de lissage exponentiel (EWM) avec différents paramètres alpha
for i in np.arange(0.2, 1, 0.2):
    bitcoin.loc['2019-09', 'Close'].ewm(alpha=i).mean().plot(label=f'EWM {i}', ls='--', alpha=0.7)

# Personnalisation du graphique
plt.legend()
plt.title("Lissage exponentiel du prix du Bitcoin - Septembre 2019")
plt.xlabel("Date")
plt.ylabel("Prix de clôture (EUR)")
plt.grid(True, linestyle="--", alpha=0.5)

# Affichage du graphique
plt.show()


#Partie avec l'ethereum 
file_path = r"C:\Users\enzoc\Desktop\Data_Analysis_Python\data_set\ETH-EUR.csv"
try:    
    ethereum = pd.read_csv(file_path, parse_dates=True, index_col='Date')
    print("Fichier chargé avec succès !")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable.")
    exit()
    
# --- AFFICHAGE DU PRIX DE L'ETHEREUM EN 2019 ---
plt.figure(figsize=(12, 6))
ethereum.loc['2019', 'Close'].plot(label="Ethereum", color="orange")
plt.title("Évolution du prix de l'Ethereum en 2019")
plt.xlabel("Date")
plt.ylabel("Prix de clôture (EUR)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# --- FUSION DES DATAFRAMES ---
print("\n🔄 Fusion des DataFrames Bitcoin et Ethereum...")

# Fusion avec INNER JOIN (Dates communes uniquement)
btc_eth_inner = pd.merge(bitcoin, ethereum, on='Date', how='inner', suffixes=('_btc', '_eth'))
print("\nDataFrame combiné (INNER JOIN) :")
print(btc_eth_inner.head())

# Fusion avec OUTER JOIN (Toutes les dates, même sans correspondance)
btc_eth_outer = pd.merge(bitcoin, ethereum, on='Date', how='outer', suffixes=('_btc', '_eth'))
print("\nDataFrame combiné (OUTER JOIN) :")
print(btc_eth_outer.head())

# --- VISUALISATION DES PRIX BTC ET ETH ---
plt.figure(figsize=(12, 8))

# Affichage des prix BTC et ETH sur des sous-graphiques
btc_eth_inner[['Close_btc', 'Close_eth']].plot(subplots=True, figsize=(12, 8), title="Comparaison des prix BTC vs ETH")
plt.tight_layout()
plt.show()

# --- CALCUL DE LA CORRÉLATION ENTRE BTC ET ETH ---
correlation_matrix = btc_eth_inner[['Close_btc', 'Close_eth']].corr()
print("\n📊 Matrice de corrélation entre BTC et ETH :")
print(correlation_matrix)

#Exercie strategie de la tortue
# Initialisation des colonnes Buy et Sell
bitcoin['Buy'] = np.zeros(len(bitcoin))
bitcoin['Sell'] = np.zeros(len(bitcoin))

# Boucle pour appliquer la stratégie de la tortue
for i in range(28, len(bitcoin)):  # On commence à 28 pour éviter NaN dans rolling()
    max_28 = bitcoin['Close'].rolling(window=28).max().iloc[i]
    min_28 = bitcoin['Close'].rolling(window=28).min().iloc[i]
    current_price = bitcoin['Close'].iloc[i]

    if current_price > max_28:
        bitcoin.at[i, 'Buy'] = 1  # Signal d'achat
    elif current_price < min_28:
        bitcoin.at[i, 'Sell'] = -1  # Signal de vente

# Affichage des premières lignes pour vérifier
print(bitcoin[['Close', 'Buy', 'Sell']].head(40))

