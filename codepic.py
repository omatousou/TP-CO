import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Chargement des données
hres = np.loadtxt("sp250312_MR0025_APT05_cospinu_gaz_296K_CO_0p1T_Strong4T_0p8trou0000.txt", delimiter=",")

# Sélection des données dans l'intervalle [2050, 2300]
mask = (hres[:, 0] >= 2050) & (hres[:, 0] <= 2250)
filtered_data = hres[mask]

x_fit = filtered_data[:, 0]  # Longueurs d'onde dans l'intervalle [2050, 2250]
y_fit = filtered_data[:, 1]  # Intensités correspondantes

coefficients = np.polyfit(x_fit, y_fit, 1)  # Fit linéaire

# Les coefficients sont dans l'ordre [a, b]
a, b = coefficients

# recentrage en 0 fonction lineaire vers affine
filtered_data[:,1] = filtered_data[:,1] - b

# Matrice de Rotation 
theta = np.arcsin(a)
filtered_data = np.dot(np.array([[np.cos(-theta),-np.sin(-theta)],[np.sin(-theta),np.cos(-theta)]]),np.transpose(filtered_data))

filtered_data = np.transpose(filtered_data)

# Détection des pics (maxima) avec des paramètres ajustés
peaks, properties = find_peaks(
    filtered_data[:, 1], 
    height=0.025,        # Seuil minimum d'intensité pour être considéré comme un pic
    distance=1,          # Distance minimale entre deux pics (ajuster selon les données)
)

# Extraction des coordonnées des pics
peaks_data = filtered_data[peaks]
peaks_data = np.concatenate((peaks_data, np.array([[2064, 0.031]])), axis=0) # pic non détecté, ajouté à la main

# Normalisation en probabilité (somme des intensités = 1)
normalized_intensities = (peaks_data[:, 1]) / np.sum(peaks_data[:, 1])

# Sauvegarde dans un fichier
np.savetxt("peaks_detected.txt", peaks_data, fmt="%.6f", delimiter=",", header="Wavelength, Intensity", comments="")

# Création des figures
fig, axs = plt.subplots(2, 1, figsize=(8, 10), dpi=300)  # Deux graphes empilés

# Premier graphe : Spectre complet avec les pics marqués
axs[0].plot(filtered_data[:, 0], filtered_data[:, 1], label="Spectre")
axs[0].scatter(peaks_data[:, 0], peaks_data[:, 1], color='red', label="Pics détectés", zorder=3)

axs[0].set_xlabel("Longueur d'onde (nm)")
axs[0].set_ylabel("Intensité")
axs[0].legend()
axs[0].set_xlim([2050, 2250])
axs[0].set_ylim([-0.1, 0.5])
axs[0].invert_xaxis()  # Optionnel

# Deuxième graphe : Diracs normalisés en probabilité (avec stem)
markerline, stemlines, baseline = axs[1].stem(
    peaks_data[:, 0], 
    normalized_intensities, 
    linefmt='-b',  # Lignes bleues
    markerfmt='ro',  # Marqueurs rouges de forme 'o' (cercle)
    basefmt=" ",  # Pas de base, pas de ligne horizontale
    label="Pics détectés"
)

# Modifier la taille des marqueurs
markerline.set_markersize(5)  # Ajuster la taille des marqueurs


axs[1].set_xlabel("Longueur d'onde (nm)")
axs[1].set_ylabel("Probabilité")
axs[1].set_title("Spectre avec uniquement les pics détectés (normalisé)")
axs[1].legend()
axs[1].invert_xaxis()  # Optionnel

# Ajouter le label du fit
axs[0].legend()

# Affichage des graphes
plt.tight_layout()
plt.show()

print(f"Nombre de pics détectés : {len(peaks_data)}")
print("Les coordonnées des pics ont été enregistrées dans 'peaks_detected.txt'.")
