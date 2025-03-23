import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Chargement des données depuis un fichier texte
hres = np.loadtxt("sp250312_MR0025_APT05_cospinu_gaz_296K_CO_0p1T_Strong4T_0p8trou0000.txt", delimiter=",")

# Sélection des données dans l'intervalle de longueurs d'onde [2050, 2300]
mask = (hres[:, 0] >= 2050) & (hres[:, 0] <= 2250)  # Création d'un masque pour filtrer les données
filtered_data = hres[mask]  # Application du masque aux données

# Extraction des longueurs d'onde et des intensités
x_fit = filtered_data[:, 0]  # Longueurs d'onde dans l'intervalle [2050, 2250]
y_fit = filtered_data[:, 1]  # Intensités correspondantes

# Réalisation d'un ajustement linéaire (polynôme de degré 1)
coefficients = np.polyfit(x_fit, y_fit, 1)  # Calcul des coefficients du polynôme linéaire
a, b = coefficients  # Coefficients a (pente) et b (ordonnée à l'origine)

# Recentrement des données autour de 0 en soustrayant la fonction linéaire
filtered_data[:, 1] = filtered_data[:, 1] - b  # Soustraction de l'ordonnée à l'origine pour recentrer

# Matrice de rotation pour appliquer une rotation affine en fonction de la pente 'a'
theta = np.arcsin(a)  # Calcul de l'angle de rotation à partir de la pente
rotation_matrix = np.array([[np.cos(-theta), -np.sin(-theta)], [np.sin(-theta), np.cos(-theta)]])  # Matrice de rotation
filtered_data = np.dot(rotation_matrix, np.transpose(filtered_data))  # Application de la rotation aux données
filtered_data = np.transpose(filtered_data)  # Retourner les données pour les rendre à la forme d'origine

# Détection des pics (maxima) dans les données filtrées
peaks, properties = find_peaks(
    filtered_data[:, 1], 
    height=0.025,        # Seuil minimum d'intensité pour être considéré comme un pic
    distance=1,          # Distance minimale entre deux pics (ajuster selon les données)
)

# Extraction des coordonnées des pics détectés
peaks_data = filtered_data[peaks]  # Récupération des données des pics
peaks_data = np.concatenate((peaks_data, np.array([[2064, 0.031]])), axis=0)  # Ajout manuel d'un pic non détecté

# Normalisation des intensités en probabilité (somme des intensités = 1)
normalized_intensities = (peaks_data[:, 1]) / np.sum(peaks_data[:, 1])  # Normalisation des intensités

# Sauvegarde des données des pics détectés dans un fichier texte
np.savetxt("peaks_detected.txt", peaks_data, fmt="%.6f", delimiter=",", header="Wavelength, Intensity", comments="")

# Création des figures pour afficher les résultats
fig, axs = plt.subplots(2, 1, figsize=(8, 10), dpi=300)  # Création de deux graphes empilés

# Premier graphe : Spectre complet avec les pics marqués
axs[0].plot(filtered_data[:, 0], filtered_data[:, 1], label="Spectre")  # Tracé du spectre
axs[0].scatter(peaks_data[:, 0], peaks_data[:, 1], color='red', label="Pics détectés", zorder=3)  # Marquage des pics détectés

# Paramétrage du premier graphe
axs[0].set_xlabel("Longueur d'onde (nm)")  # Label de l'axe des abscisses
axs[0].set_ylabel("Intensité")  # Label de l'axe des ordonnées
axs[0].legend()  # Affichage de la légende
axs[0].set_xlim([2050, 2250])  # Limites de l'axe des x
axs[0].set_ylim([-0.1, 0.5])  # Limites de l'axe des y
axs[0].invert_xaxis()  # Inversion de l'axe des x (optionnel)

# Deuxième graphe : Représentation des pics sous forme de Diracs normalisés en probabilité
axs[1].stem(
    peaks_data[:, 0], 
    normalized_intensities, 
    linefmt='-b',  # Lignes bleues
    markerfmt='ro',  # Marqueurs rouges de forme 'o' (cercle)
    basefmt=" ",  # Pas de base, pas de ligne horizontale
    label="Pics détectés", 
    use_line_collection=True
)

# Changer la taille des marqueurs (paramétrage des marqueurs)
markersize = 5  # Taille des marqueurs
axs[1].collections[1].set_sizes([markersize] * len(peaks_data))  # Modification de la taille des marqueurs

# Paramétrage du deuxième graphe
axs[1].set_xlabel("Longueur d'onde (nm)")  # Label de l'axe des abscisses
axs[1].set_ylabel("Probabilité")  # Label de l'axe des ordonnées
axs[1].set_title("Spectre avec uniquement les pics détectés (normalisé)")  # Titre du graphique
axs[1].legend()  # Affichage de la légende
axs[1].invert_xaxis()  # Inversion de l'axe des x (optionnel)

# Ajouter la légende du premier graphe
axs[0].legend()

# Affichage des graphes
plt.tight_layout()  # Ajuste l'espacement pour une meilleure lisibilité
plt.show()  # Affiche les graphes

# Affichage du nombre de pics détectés et de l'emplacement du fichier de sauvegarde
print(f"Nombre de pics détectés : {len(peaks_data)}")
print("Les coordonnées des pics ont été enregistrées dans 'peaks_detected.txt'.")
