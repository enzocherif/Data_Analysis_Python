import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Chargement des données avec gestion d'erreur en cas de fichier introuvable
file_path = r"C:\Users\enzoc\Desktop\Data_Analysis_Python\data_set\titanic3 (3).xls"
try:
    data = pd.read_excel(file_path)
    print("Fichier chargé avec succès !")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{file_path}' est introuvable.")
    exit()

# Affichage des informations de base sur le dataset
print("Dimensions du dataset :", data.shape)
print("Colonnes disponibles :", data.columns)
print("Aperçu des premières lignes :")
print(data.head())

# Suppression des colonnes non pertinentes pour notre analyse
columns_to_drop = ['name', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 
                   'embarked', 'boat', 'body', 'home.dest']
data = data.drop(columns=columns_to_drop, axis=1)

print("\nColonnes après suppression :", data.columns)
print("Aperçu des premières lignes après nettoyage :")
print(data.head())

# Analyse statistique du dataset
print("\nStatistiques générales du dataset :")
print(data.describe())

# Gestion des valeurs manquantes
# On pourrait remplir les valeurs manquantes avec la moyenne, mais cela altérerait les données
# On préfère donc supprimer les lignes avec des valeurs manquantes
data = data.dropna(axis=0)

print("\nDataset après suppression des lignes avec valeurs manquantes :")
print(f"Nombre de lignes après nettoyage : {len(data)}")  # Avant : 1309, après : 1046
print(data.describe())  # Nouvelle analyse après suppression

# Analyse de la répartition des classes sociales des passagers
print("\nRépartition des passagers par classe :")
print(data['pclass'].value_counts())

# Visualisation de la répartition des classes
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
data['pclass'].value_counts().plot(kind='bar', color=['blue', 'green', 'red'])
plt.xlabel("Classe")
plt.ylabel("Nombre de passagers")
plt.title("Répartition des passagers par classe")

# Visualisation de la distribution des âges
plt.subplot(1, 2, 2)
data['age'].hist(bins=20, color='orange', edgecolor='black')
plt.xlabel("Âge")
plt.ylabel("Nombre de passagers")
plt.title("Distribution des âges des passagers")

plt.tight_layout()
plt.show()

# Analyse des moyennes en fonction du sexe et de la classe
print("\nMoyenne des caractéristiques selon le sexe et la classe :")
print(data.groupby(['sex', 'pclass']).mean())

# Filtrage des passagers de moins de 18 ans
print("\nAperçu des âges des 10 premiers passagers :")
print(data['age'][0:10])

# Vérification des mineurs
print("\nPassagers mineurs (moins de 18 ans) :")
print(data[data['age'] < 18]['pclass'].value_counts())

# Analyse des mineurs par sexe et classe
print("\nMoyenne des caractéristiques des mineurs par sexe et classe :")
print(data[data['age'] < 18].groupby(['sex', 'pclass']).mean())

# Exploration des indexations avec .iloc et .loc
print("\nExemples d'accès aux données :")
print("Première valeur du dataset :", data.iloc[0, 0])
print("Sous-ensemble avec .iloc :")
print(data.iloc[0:2, 0:2])
print("Sous-ensemble avec .loc :")
print(data.loc[0:2, ['age', 'sex']])

# --- FILTRAGE DES PASSAGERS SELON L'ÂGE ---

# Cas 1 : Passagers de moins de 20 ans
data1 = data[data['age'] < 20]
print(f"\nPassagers de moins de 20 ans : {len(data1)}")
print(data1.head())

# Cas 2 : Passagers âgés de 20 à 30 ans
data2 = data[(data['age'] >= 20) & (data['age'] < 30)]
print(f"\nPassagers âgés de 20 à 30 ans : {len(data2)}")
print(data2.head())

# Cas 3 : Passagers âgés de 30 à 40 ans
data3 = data[(data['age'] >= 30) & (data['age'] < 40)]
print(f"\nPassagers âgés de 30 à 40 ans : {len(data3)}")
print(data3.head())

# Cas 4 : Passagers de plus de 40 ans
data4 = data[data['age'] >= 40]
print(f"\nPassagers de plus de 40 ans : {len(data4)}")
print(data4.head())

# --- VISUALISATION DE LA RÉPARTITION DES ÂGES ---

plt.figure(figsize=(12, 6))

# Histogramme de la répartition des âges
plt.hist(data['age'], bins=[0, 20, 30, 40, 50, 60, 70, 80], edgecolor='black', alpha=0.7, color='skyblue')
plt.axvline(x=20, color='r', linestyle='dashed', linewidth=1, label="20 ans")
plt.axvline(x=30, color='g', linestyle='dashed', linewidth=1, label="30 ans")
plt.axvline(x=40, color='b', linestyle='dashed', linewidth=1, label="40 ans")
plt.xlabel("Âge")
plt.ylabel("Nombre de passagers")
plt.title("Répartition des passagers par tranche d'âge")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

# --- STATISTIQUES PAR TRANCHE D'ÂGE ---

print("\nMoyenne des caractéristiques pour chaque tranche d'âge :")
print("\nMoins de 20 ans :")
print(data1.describe())

print("\n20 à 30 ans :")
print(data2.describe())

print("\n30 à 40 ans :")
print(data3.describe())

print("\nPlus de 40 ans :")
print(data4.describe())