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




