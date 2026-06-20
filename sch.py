import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt

# ====================================================================
# 1.a + 1.b : derivee premiere 
# ===================================================================
def derivee_premiere(f, dx):
    """
    calcule de derivee premiere d'un tableau f avec pas d'espace dx.
    use de la méthode des differences finies centrees.
    """
    df = np.zeros(len(f))
    
    # pour points intérieurs (du 1er au point n-2)
    for i in range(1, len(f) - 1):
        df[i] = (f[i+1] - f[i-1]) / (2 * dx)
        
    # gere les bords pour pas sortir du tab
    df[0] = (f[1] - f[0]) / dx
    df[-1] = (f[-1] - f[-2]) / dx
    
    return df

# =====================================================================
# 2. Derivee seconde
# ===================================================================
def derivee_seconde(f, dx):
    """
    calcule derivee seconde d'un tableau f avec un pas d'espace dx.
    formule centrée : (f[i+1] - 2*f[i] + f[i-1]) / dx^2
    """
    d2f = np.zeros(len(f))
    
    for i in range(1, len(f) - 1):
        d2f[i] = (f[i+1] - 2 * f[i] + f[i-1]) / (dx**2)
        
    # gére les bords
    d2f[0] = d2f[1]
    d2f[-1] = d2f[-2]
    
    return d2f

# ====================================================================
# 1.c & 1.d : fonctions de test 
# ===================================================================
def f_carre(x):
    return x**2

def f_derivee_carree(x):
    return 2 * x

if __name__ == "__main__":
    # paramètres de la grille numérique
    npts = 100
    x = np.linspace(-5, 5, npts)
    dx = x[1] - x[0] # calcul du pas entre deux points consécutifs
    
    # calcul des fonctions
    y = f_carre(x)
    y_prime_carree = f_derivee_carree(x)
    
    # calcul numérique 
    y_prime_numerique = derivee_premiere(y, dx)
    y_seconde_numerique = derivee_seconde(y, dx) 
    
    # 1.e : calcul de l'erreur relative 
    erreur_relative = np.zeros_like(x)
    masque = y_prime_carree != 0
    erreur_relative[masque] = np.abs((y_prime_numerique[masque] - y_prime_carree[masque]) / y_prime_carree[masque])

    print(f"erreur relative maximale constatée : {np.max(erreur_relative):.2e}")

    # ---affichage Graphique --
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # premier graphique : Dérivée première
    ax1.plot(x, y_prime_carree, 'b-', label="Théorie : $2x$", lw=2)
    ax1.plot(x, y_prime_numerique, 'ro', markevery=4, label="Numérique : algorithme", alpha=0.7)
    ax1.set_title("Validation de l'algorithme de dérivation première")
    ax1.set_ylabel("y'")
    ax1.grid(True, linestyle=":")
    ax1.legend()
    
    # Deuxième graphique : Dérivée seconde
    ax2.plot(x, np.ones_like(x) * 2, 'g-', label="Théorie : $2$", lw=2)
    ax2.plot(x, y_seconde_numerique, 'mo', markevery=4, label="Numérique : dérivée seconde", alpha=0.7)
    ax2.set_title("Validation de la dérivée seconde")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y''")
    ax2.grid(True, linestyle=":")
    ax2.legend()
    ax2.set_ylim(0, 4)
    plt.tight_layout()
    plt.show()


# ====================================================================
# 3.2.1 : EQUATION DE SCHRODINGER 1D, POTENTIEL CONSTANT V0
#   i hbar dpsi/dt = -hbar^2/(2m) d2psi/dx2 + V0 psi
# ====================================================================
HBAR = 1.0
MASS = 1.0
V0   = 0.0
 
# 3.2.3 : grilles espace et temps (numpy.linspace)
NX = 2000
X_MIN, X_MAX = -20.0, 100.0
x  = np.linspace(X_MIN, X_MAX, NX)
dx = x[1] - x[0]
 
NT    = 15000
T_MAX = 20.0
t     = np.linspace(0, T_MAX, NT)
dt    = t[1] - t[0]
 
# 3.2.2 : fonction d'onde = TABLEAU 2D (nx lignes, nt colonnes)
#         1re colonne = paquet initial ; le reste = vide (empty)
psi = np.empty((NX, NT), dtype=complex)
 
# --- paquet gaussien initial (convention accordee au prof : a = 1/SIGMA_prof) ---
K0 = 1.0
X0 = -20.0
SIGMA_PROF = 0.2
a        = 1.0 / SIGMA_PROF
sigma_x0 = a / np.sqrt(2)
psi0 = np.exp(-((x - X0)**2)/(4*sigma_x0**2)) * np.exp(1j*K0*x)
psi0 /= np.sqrt(simpson(np.abs(psi0)**2, x))
psi[:, 0] = psi0                      # premiere colonne remplie
 
# 3.2.4 : algorithme combinant derivee 1ere en TEMPS + 2nde en ESPACE
#   psi[:,n+1] = psi[:,n] + dt*(-i/hbar)*( -hbar^2/(2m) d2psi/dx2 + V0 psi )
inv_ihbar   = -1j * dt / HBAR
coeff_deriv = -(HBAR**2) / (2*MASS*dx**2)
 
print("Evolution temporelle (remplissage colonne par colonne)...")
for n in range(NT - 1):
    col = psi[:, n]
    nouvelle = np.zeros(NX, dtype=complex)
    nouvelle[1:-1] = col[1:-1] + inv_ihbar*(
        coeff_deriv*(col[2:] - 2*col[1:-1] + col[:-2]) + V0*col[1:-1]
    )
    nouvelle[0] = 0.0; nouvelle[-1] = 0.0
    psi[:, n+1] = nouvelle
    # arret si explosion (sinon overflow inutile)
    if n % 50 == 0 and simpson(np.abs(psi[:,n+1])**2, x) > 1e8:
        print(f"  -> norme divergente des t = {(n+1)*dt:.3f}")
        n_stop = n+1
        break
else:
    n_stop = NT-1
print("Fin.\n")
 
# ====================================================================
# 3.2.5 : CONFRONTATION avec la théorie (V0 = 0)
# ====================================================================

def norme(col):
    return simpson(np.abs(col)**2, x)

def position_max(col):
    densite = np.abs(col)**2
    return x[np.argmax(densite)]

def position_theorique(temps):
    vg = HBAR * K0 / MASS
    return X0 + vg * temps

print("\n--- 3.2.5 : CONFRONTATION AVEC LA THEORIE ---")

for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:

    idx = int(frac * n_stop)

    temps = t[idx]

    norme_num = norme(psi[:, idx])

    x_num = position_max(psi[:, idx])

    x_theo = position_theorique(temps)

    erreur = abs(x_num - x_theo)

    print(f"\nTemps t = {temps:.3f}")
    print(f"Norme numérique       : {norme_num:.6e}")
    print(f"Position numérique    : {x_num:.6f}")
    print(f"Position théorique    : {x_theo:.6f}")
    print(f"Erreur sur position   : {erreur:.6f}")
