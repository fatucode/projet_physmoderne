# ====================================================================
# ÉTUDE DE LA SUPERPOSITION D'ONDES PLANES ET PHÉNOMÈNE DE BATTEMENT
# Ce script simule la somme d'un nombre restreint d'ondes planes
# pour observer la formation d'un paquet d'ondes (localisation spatiale).
# ====================================================================

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# 1. PARAMÈTRES PHYSIQUES ET CONFIGURATION DE LA SIMULATION
# --------------------------------------------------------------------

K_CENTRE = 1.0          # Nombre d'onde central (fixe la périodicité principale) [1/m]
DELTA_K  = K_CENTRE / 10 # Écart constant entre les différents nombres d'ondes
N_WAVES  = 3            # Nombre d'ondes planes à superposer (choix entre 1 et 9)

# Définition d'un spectre discret de nombres d'ondes (valeurs de k)
# Ces valeurs spécifiques permettent d'observer un motif d'interférence net.
LISTE_K = [
    K_CENTRE - 1.789 * DELTA_K,
    K_CENTRE - 1.200 * DELTA_K,
    K_CENTRE - 1.000 * DELTA_K,
    K_CENTRE,
    K_CENTRE + 1.000 * DELTA_K,
    K_CENTRE + 1.200 * DELTA_K,
    K_CENTRE + 1.789 * DELTA_K,
    K_CENTRE + 2.050 * DELTA_K,
    K_CENTRE - 2.103 * DELTA_K,
][:N_WAVES] # On ne conserve que les N_WAVES premières ondes

# Définition de l'espace et du temps
# Pour observer une cloche symétrique, la grille spatiale doit être centrée sur x = 0.
X = np.linspace(-25, 25, 500)  # Grille de 500 points allant de -25m à +25m
T = 0.0                        # On fige le temps à t = 0 pour que toutes les ondes soient en phase à l'origine

# --------------------------------------------------------------------
# 2. MODÉLISATION MATHÉMATIQUE DES ONDES
# --------------------------------------------------------------------

def compute_omega(k: float, speed: float = 1.0, dispersion: str = "linear") -> float:
    """
    Calcule la pulsation temporelle (omega) via la relation de dispersion.
    - Mode 'linear' : Ondes classiques (ex: acoustique, lumière dans le vide) où omega = c * |k|
    - Mode 'schrodinger' : Ondes quantiques où omega = hbar * k^2 / (2 * m)
    """
    if dispersion == "linear":
        return speed * abs(k)
    elif dispersion == "schrodinger":
        hbar, mass = 1.0, 1.0 # Utilisation du système d'unités réduites
        return hbar * k**2 / (2 * mass)
    raise ValueError(f"Dispersion inconnue : {dispersion}")

def compute_plane_wave(k: float, x: np.ndarray, t: float) -> np.ndarray:
    """
    Calcule l'amplitude complexe d'une onde plane monochromatique unidimensionnelle.
    Formule mathématique : Psi(x,t) = exp( i * (omega * t - k * x) )
    """
    omega = compute_omega(k)
    phase = 1j * (omega * t - k * x) # 1j représente le nombre imaginaire complexe 'i'
    return np.exp(phase)

# --------------------------------------------------------------------
# 3. EXÉCUTION DES CALCULS ET DIAGNOSTICS EN CONSOLE
# --------------------------------------------------------------------

print(f"--- SIMULATION : SUPERPOSITION DE {N_WAVES} ONDES PLANES ---")
print(f"Grille spatiale : de {X.min()}m à {X.max()}m | Instant d'observation t = {T}s\n")

# Tableau à deux dimensions pour stocker l'onde complexe de chaque composante
# Ligne = indice de l'onde, Colonne = valeur spatiale en x
waves = np.zeros((N_WAVES, len(X)), dtype=complex)
liste_labels = []

for i, k in enumerate(LISTE_K):
    # Calcul de la contribution de la i-ème onde plane
    waves[i] = compute_plane_wave(k, X, T)
    
    # Calcul des grandeurs caractéristiques physiques pour l'explication
    wavelength = 2 * np.pi / abs(k) # Longueur d'onde lambda = 2*pi / |k|
    n_osc = (X.max() - X.min()) / wavelength # Nombre d'oscillations visibles sur la grille
    
    label = f"Onde {i+1} : Nombre d'onde k={k:.2f} rad/m | Longueur d'onde \u03bb={wavelength:.2f} m"
    liste_labels.append(label)
    print(label)

# --------------------------------------------------------------------
# 4. CONSTRUCTION DE LA REPRÉSENTATION GRAPHIQUE (2 GRAPHES SUPERPOSÉS)
# --------------------------------------------------------------------

# Création d'une figure à deux sous-graphiques partageant le même axe horizontal X
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Analyse Spectrale : De l'Onde Plane Monochromatique au Paquet d'Ondes", fontsize=13, fontweight='bold')

# Définition d'une charte de couleurs (palette plasma) pour différencier les signaux
colors = plt.cm.plasma(np.linspace(0.1, 0.75, N_WAVES))

# --- GRAPHE SUPÉRIEUR : LES COMPOSANTES INDIVIDUELLES ---
# On affiche la partie réelle de chaque onde plane séparément.
# Une onde plane seule a une amplitude constante (ici égale à 1) sur tout l'espace.
for i in range(N_WAVES):
    ax1.plot(X, np.real(waves[i]), color=colors[i], lw=1.2, label=liste_labels[i])
ax1.set_title("1. Composantes fréquentielles individuelles (Partie Réelle)", fontsize=11, loc='left')
ax1.set_ylabel("Amplitude $Re(\Psi_i)$")
ax1.grid(True, ls=":", alpha=0.6)
ax1.legend(loc="upper right", fontsize=9)

# --- GRAPHE INFÉRIEUR : LA SUPERPOSITION FINALE (BATTEMENT) ---
# Somme algébrique de toutes les ondes : Principe de superposition
superposition = np.real(np.sum(waves, axis=0))
ax2.plot(X, superposition, color="crimson", lw=2, label=f"Signal résultant (Somme des {N_WAVES} ondes)")

# Calcul mathématique de l'enveloppe de modulation (frontière théorique des interférences)
# Elle correspond au module de la somme des phases spatiales complexes.
enveloppe = np.abs(np.sum(np.exp(-1j * np.array(LISTE_K)[:, None] * X), axis=0))

# Tracé des lignes d'enveloppe supérieure et inférieure (lignes pointillées noires)
ax2.plot(X, enveloppe, "k--", lw=1.2, alpha=0.7, label="Enveloppe théorique du battement")
ax2.plot(X, -enveloppe, "k--", lw=1.2, alpha=0.7)

ax2.set_title("2. Paquet d'ondes résultant par interférence constructive et destructive", fontsize=11, loc='left')
ax2.set_xlabel("Position spatiale $x$ [m]")
ax2.set_ylabel("Amplitude Totale $\Sigma \Psi_i$")
ax2.grid(True, ls=":", alpha=0.6)
ax2.legend(loc="upper right", fontsize=9)

# Ajustement automatique des marges pour éviter les chevauchements
plt.tight_layout()

# Affichage de la fenêtre graphique
plt.show()
