
import matplotlib.pyplot as plt
import numpy as np

Te = 0
we = 2169.81358
wexe = 13.28831
weye = 1.04109 * 10**(-2)
Be = 1.93128087
ale = 1.750441 * 10**(-2)
De = 6.12147 * 10**(-6)
gamae = 0
betae = 0
T = 296
Kb = 1.380649 * 10**(-23)
c = 299792458*100
h = 6.62607015 * 10**(-34)



def Energie0(J):
    v = 0
    return Te + we * (v + 1 / 2) - wexe * (v + 1 / 2)**2 + weye * (v + 1 / 2)**3 + \
           (Be - ale * (v + 1 / 2) + gamae * (v + 1 / 2)**2) * (J * (J + 1)) - \
           (De - betae * (v + 1 / 2)) * (J * (J + 1))**2

def Energie1(J):
    w = 1
    return Te + we * (w + 1 / 2) - wexe * (w + 1 / 2)**2 + weye * (w+ 1 / 2)**3 + \
           (Be - ale * (w + 1 / 2) + gamae * (w + 1 / 2)**2) * (J * (J + 1)) - \
           (De - betae * (w + 1 / 2)) * (J * (J + 1))**2




Jm = 32


siP = []
siR = []
plt.figure(dpi = 300)
for i in range(-Jm, Jm+1):
    plt.scatter(i,Energie0(np.abs(i)), color='blue',s = 1)
    plt.plot([0,i],[Energie0(np.abs(i)),Energie0(np.abs(i))],'b',alpha = 0.1)
    
    plt.plot([0,i],[Energie1(np.abs(i)),Energie1(np.abs(i))],'r',alpha = 0.1)
    plt.scatter(i,Energie1(np.abs(i)),color='red',s=1)
    Energie0(np.abs(i))
plt.show()

# Calcul de siP et siR
for i in range(0, Jm+1):
    siP.append([i, -(Energie0(i) - Energie1(i + 1))])
    siR.append([i, -(Energie0(i) - Energie1(i - 1))])







# Somme pour le calcul de roi
somme = 0
for i in range(0, Jm + 1):
    somme += (2 * i + 1) * np.exp(-Energie0(i) * h * c / (Kb * T))

def roi(j):
    return (2 * j + 1) * np.exp(-Energie0(j) * h * c / (Kb * T)) / somme

# Calcul de Roi
Roi = []
for i in range(0, Jm + 1):
    Roi.append(roi(i))

# Graphique de Roi en fonction de siR
plt.figure(dpi=300)
siR_values = [item[1] for item in siR]  # extraire les valeurs de siR

markerline, stemlines, baseline = plt.stem(
    siR_values, Roi[:len(siR_values)],
    linefmt='-',
    markerfmt='ro',  # Marqueurs rouges de forme 'o' (cercle)
    basefmt=" ",  # Pas de base, pas de ligne horizontale
    label="Roi en fonction de siR"
)

# Modifier la taille des marqueurs
markerline.set_markersize(2)  # Ajuster la taille des marqueurs
markerline.set_color('b')  # Ajuster la taille des marqueurs


# Graphique de Roi en fonction de siP
siP_values = [item[1] for item in siP]  # extraire les valeurs de siP


markerline, stemlines, baseline = plt.stem(
    siP_values, Roi[:len(siP_values)],
    linefmt='-',
    markerfmt='ro',  # Marqueurs rouges de forme 'o' (cercle)
    basefmt=" ",  # Pas de base, pas de ligne horizontale
    label="Roi en fonction de siP"
)

# Modifier la taille des marqueurs
markerline.set_markersize(2)  # Ajuster la taille des marqueurs
markerline.set_color('r')  # Ajuster la taille des marqueurs
stemlines.set_color('orange') 

plt.gca().invert_xaxis()
plt.xlabel('siP / siR')
plt.ylabel('Roi')
plt.xlim([2000,2300])
plt.ylim([0,0.10])
plt.legend()
plt.show()








# Chargement des données
peaks_data = np.loadtxt("peaks_detected.txt",delimiter=",")
#peaks_data = np.loadtxt("COTest1.txt")


# Normalisation des probabilités
normalized_intensities =( peaks_data[:, 1]) / np.sum(( peaks_data[:, 1]))

# Tracé de l'histogramme

# Deuxième graphe : Histogramme normalisé en probabilité
plt.figure(dpi=300)


markerline, stemlines, baseline = plt.stem(
    peaks_data[:, 0], normalized_intensities,
    linefmt='-',
    markerfmt='ro',  # Marqueurs rouges de forme 'o' (cercle)
    basefmt=" ",  # Pas de base, pas de ligne horizontale
    label="Pics détectés"
)

# Modifier la taille des marqueurs
markerline.set_markersize(2)  # Ajuster la taille des marqueurs
markerline.set_color('r')  # Ajuster la taille des marqueurs
stemlines.set_color('b') 



plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Probabilité")
plt.title("Spectre avec uniquement les pics détectés (normalisé)")
plt.gca().invert_xaxis() # Optionnel

# Affichage des graphes
plt.tight_layout()
plt.show()


# Normalisation des probabilités de Roi
normalized_Roi = np.array(Roi) / np.sum(Roi)  # Normalisation pour que la somme fasse 1


# Mise à l'échelle pour avoir la même amplitude maximale que Roi
scale_factor = np.max(normalized_Roi) / np.max(normalized_intensities)
normalized_intensities *= scale_factor

# Taille de la figure
plt.figure(figsize=(12, 7),dpi=300)

# Histogramme de Roi en fonction de siR
plt.bar(siR_values, normalized_Roi[:len(siR_values)], width=3, 
        label='Roi en fonction de siR', alpha=0.7, color='blue', edgecolor='black')

# Histogramme de Roi en fonction de siP
plt.bar(siP_values, normalized_Roi[:len(siP_values)], width=3, 
        label='Roi en fonction de siP', alpha=0.7, color='green', edgecolor='black')

# Ajout des pics détectés avec mise à l'échelle
plt.bar(peaks_data[:, 0], normalized_intensities, width=3, 
        color='red', alpha=0.6, edgecolor='black', label="Pics détectés", zorder=3)

# Ajustements graphiques
plt.gca().invert_xaxis()  # Inverser l'axe des x pour correspondre aux conventions spectrales
plt.xlabel('SiP et SiR', fontsize=14)
plt.ylabel('Probabilité normalisée', fontsize=14)
plt.xlim([2000, 2250])
plt.ylim([0, max(np.max(normalized_Roi), np.max(normalized_intensities)) * 1.2])  # Ajustement de la visibilité
plt.legend(fontsize=12)
plt.title("Superposition des distributions normalisées avec meilleure visibilité", fontsize=16)

# Grille pour meilleure lisibilité
plt.grid(True, linestyle='--', alpha=0.5)

# Affichage
plt.tight_layout()
plt.show()