# ====================================================================
# PARTIE 4 : ÉTUDE DE L'EFFET TUNNEL QUANTIQUE
# - Observation de l'impact de la largeur 'a' et de la hauteur 'V0'
# - Mesure du temps de franchissement et calcul du retard (Hartman)
# - Tracé de la fonction d'onde à 4 moments clés du choc
# ====================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Sécurité pour les versions de scipy : utilise np.trapezoid pour intégrer
trapz = np.trapezoid         

# --------------------------------------------------------------------
# PARAMÈTRES DU SYSTÈME ET DE LA GRILLE NUMÉRIQUE
# --------------------------------------------------------------------
HBAR = MASS = 1.0              # On pose hbar = m = 1 (unités réduites)
NX = 5000                      # Nombre de points pour découper l'espace
x = np.linspace(-200, 200, NX) # Grille spatiale qui va de -200 à +200
dx = x[1] - x[0]               # Le pas d'espace entre deux points de la grille

# --------------------------------------------------------------------
# CARACTÉRISTIQUES DU PAQUET D'ONDES GAUSSIEN INITIAL
# --------------------------------------------------------------------
K0 = 2.0                       # Impulsion centrale de la particule
X0 = -50.0                     # Position de départ du paquet (à gauche)
SIG = 5.0                      # Largeur initiale de la cloche (étalement)
E = K0**2 / 2                  # Énergie cinétique de la particule (vaut 2.0 ici)
v_g = K0                       # Vitesse de groupe théorique (vaut 2.0 m/s)

def paquet():
    """Génère le paquet d'ondes de départ et le normalise à 1."""
    p = np.exp(-((x - X0)**2) / (4 * SIG**2)) * np.exp(1j * K0 * x)
    return p / np.sqrt(trapz(np.abs(p)**2, x))

# --------------------------------------------------------------------
# MOTEUR DE CALCUL : ALGORITHME D'ÉVOLUTION MATRICIEL
# --------------------------------------------------------------------
def make_step(V, dt):
    r = 1j * HBAR * dt / (4 * MASS * dx**2)
    s = 1j * dt / (2 * HBAR) * V
    
    # Structure de la matrice tridiagonale (3 lignes pour solve_banded)
    Ab = np.zeros((3, NX), dtype=complex)
    Ab[0, 1:] = -r       # Diagonale supérieure
    Ab[1, :] = 1 + 2*r+s # Diagonale principale
    Ab[2, :-1] = -r      # Diagonale inférieure
    
    # Application des conditions aux limites aux bords du tableau
    Ab[1, 0] = 1; Ab[0, 1] = 0; Ab[1, -1] = 1; Ab[2, -2] = 0
    
    def step(p):
        """Fait avancer la fonction d'onde p d'un pas dt."""
        b = (1 - 2*r - s) * p
        b[1:-1] += r * (p[2:] + p[:-2])
        b[0] = 0; b[-1] = 0 # On force à zéro aux limites
        return solve_banded((1, 1), Ab, b)
        
    return step

# --------------------------------------------------------------------
# FONCTIONS DE MESURE (PROBABILITÉ ET CENTRE DE MASSE TRANSMIS)
# --------------------------------------------------------------------
def cm_transmis(psi, a):
    """Calcule la position moyenne du paquet situé à droite de la barrière."""
    d = np.abs(psi)**2
    m = x > a/2 # Masque pour ne regarder que la zone après l'obstacle
    if trapz(d[m], x[m]) < 1e-9: 
        return None # Pas assez de matière transmise pour calculer
    return trapz(x[m] * d[m], x[m]) / trapz(d[m], x[m])

def proba_transmise(psi, a):
    """Calcule le coefficient de transmission T (proportion qui passe)."""
    d = np.abs(psi)**2
    m = x > a/2
    return trapz(d[m], x[m])

# --------------------------------------------------------------------
# FONCTIONS POUR CHRONOMÉTRER LES TRAJETS
# --------------------------------------------------------------------
def t_arrivee(a, V0, xref, dt=0.01, NT=20000):
    """Mesure le temps mis par le centre transmis pour franchir le point xref."""
    # Création de la barrière de potentiel centrée en x=0
    V = np.where(np.abs(x) < a/2, V0, 0.0)
    step = make_step(V, dt)
    psi = paquet() # Initialisation du paquet
    
    # Boucle temporelle
    for n in range(NT):
        psi = step(psi)
        xt = cm_transmis(psi, a)
        # Si le centre du paquet transmis dépasse la ligne d'arrivée xref
        if xt is not None and xt > xref:
            return (n + 1) * dt # On renvoie le temps exact trouvé
    return None

def transmission(a, V0, dt=0.01, NT=14000):
    """Fait tourner la simulation jusqu'à la fin pour avoir le T final."""
    V = np.where(np.abs(x) < a/2, V0, 0.0)
    step = make_step(V, dt)
    psi = paquet()
    for n in range(NT):
        psi = step(psi)
    return proba_transmise(psi, a)

# ====================================================================
# CONFIGURATION DU TEST ET AFFICHAGE DES TABLEAUX DANS LA CONSOLE
# ====================================================================
XREF = 20.0                          # Ligne d'arrivée fixe située après l'obstacle
t_libre = (XREF - X0) / v_g          # Calcul analytique du temps sans obstacle (70m / 2m/s = 35s)

print(f"Énergie particule E = {E:.1f}  |  Vitesse v_g = {v_g}  |  Ligne d'arrivée x = {XREF}\n")

# --------------------------------------------------------------------
# TABLEAU 1 : VARIATION DE LA LARGEUR 'a' (Hauteur fixée à V0 = 3.0)
# --------------------------------------------------------------------
print("4.1.c/d : Temps de franchissement selon la largeur de la barrière a (V0 = 3.0)")
print(f"{'a':>4} | {'t_avec_barr':>11} | {'t_libre':>8} | {'RETARD':>7} | {'T':>8}")
print("-"*48)
for a in [1, 2, 3, 4, 5]:
    tb = t_arrivee(a, 3.0, XREF)
    T = transmission(a, 3.0)
    if tb:
        print(f"{a:>4} | {tb:>11.2f} | {t_libre:>8.2f} | {tb-t_libre:>7.2f} | {T:>8.4f}")
    else:
        print(f"{a:>4} | {'--':>11} | {t_libre:>8.2f} | {'--':>7} | {T:>8.4f}")

# --------------------------------------------------------------------
# TABLEAU 2 : VARIATION DE LA HAUTEUR 'V0' (Largeur fixée à a = 2.0)
# --------------------------------------------------------------------
print("\n4.1.e : Temps de franchissement selon la hauteur de la barrière V0 (a = 2.0)")
print(f"{'V0':>4} | {'t_avec_barr':>11} | {'t_libre':>8} | {'RETARD':>7} | {'T':>8}")
print("-"*48)
for V0 in [2.2, 2.5, 3.0, 3.5, 4.0]:
    tb = t_arrivee(2.0, V0, XREF)
    T = transmission(2.0, V0)
    if tb:
        print(f"{V0:>4} | {tb:>11.2f} | {t_libre:>8.2f} | {tb-t_libre:>7.2f} | {T:>8.4f}")
    else:
        print(f"{V0:>4} | {'--':>11} | {t_libre:>8.2f} | {'--':>7} | {T:>8.4f}")

# ====================================================================
# GÉNÉRATION DU GRAPHIQUE UNIQUE : VISUALISATION DES ONDES
# ====================================================================
a0, V00 = 1.0, 3.0 # On choisit ces valeurs pour l'illustration
cibles = {0: None, 22: None, 30: None, 45: None} # Les 4 instants à capturer
step = make_step(np.where(np.abs(x) < a0/2, V00, 0.0), 0.01)
psi = paquet()

# Évolution temporelle globale pour enregistrer les 4 courbes
for n in range(4600):
    tc = n * 0.01
    for k in list(cibles):
        if cibles[k] is None and abs(tc - k) < 0.005:
            cibles[k] = (np.abs(psi)**2).copy() # Sauvegarde de la forme à cet instant
    psi = step(psi)

# Calcul du coefficient de transmission final réel pour le texte du graphe
T_graphe = trapz(np.abs(psi)**2 * (x > a0/2), x)

# Préparation de l'affichage avec matplotlib
nv = cibles[0].max() # Normalisation pour caler le premier pic à 1.0
inc = cibles[0]/nv; t1 = np.clip(cibles[22]/nv, 0, 1.2); t2 = cibles[30]/nv; t3 = cibles[45]/nv

fig, ax = plt.subplots(figsize=(13, 6.5))
fig.patch.set_facecolor('white')

# Dessin de la barrière de potentiel au centre (zone rouge en fond)
h = 1.3
ax.fill_between(x, 0, h, where=(np.abs(x) < a0/2), color='#d62728', alpha=0.15)
ax.vlines([-a0/2, a0/2], 0, h, color='#d62728', lw=2.5)
ax.hlines(h, -a0/2, a0/2, color='#d62728', lw=2.5)

# Tracé des 4 courbes d'ondes successives
ax.plot(x, inc, color='#1f5fd6', lw=3,   label="$t_0$ : paquet incident")
ax.plot(x, t1,  color='#ff8c00', lw=2.4, label="$t_1$ : collision (interférences)")
ax.plot(x, t2,  color='#9467bd', lw=2.4, label="$t_2$ : séparation")
ax.plot(x, t3,  color='#2ca02c', lw=3,   label="$t_3$ : état final")

# Ajout des annotations explicatives sur l'image
ax.annotate("RÉFLÉCHI", (-48, 0.42), color='#1a7a1a', fontsize=13, ha='center', fontweight='bold')
ax.annotate(f"TRANSMIS\npar effet tunnel\n(~{T_graphe*100:.0f}%)", (42, 0.20), color='#1a7a1a', 
            fontsize=12, ha='center', fontweight='bold')
ax.annotate("", xy=(33, 0.05), xytext=(42, 0.14), arrowprops=dict(arrowstyle="->", color='#1a7a1a', lw=1.8))
ax.annotate("Barrière $V_0=3$", (0, 1.35), color='#d62728', fontsize=11, ha='center', 
            fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='#d62728', alpha=0.9))

# Paramétrage des axes, légendes et titres
ax.set_xlabel("Position  $x$", fontsize=13)
ax.set_ylabel(r"Densité de probabilité  $|\Psi(x,t)|^2$  (u.a.)", fontsize=13)
ax.set_title("Effet tunnel : évolution du paquet d'ondes", fontsize=15, fontweight='bold', pad=15)
ax.set_xlim(-95, 95); ax.set_ylim(0, 1.55)
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(True, ls=":", alpha=0.4)

# Épuration du cadre graphique
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("effet_tunnel.png", dpi=140) # Enregistrement sous le nom demandé
plt.show()

print(f"\nSimulation terminée. Graphique enregistré sous 'effet_tunnel.png' (T={T_graphe:.3f})")
