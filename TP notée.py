# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
name="chien.png"
# choix possibles :
# 1ers tests (60%) : chevaux   chien
# 2emes tests (85%) : crapule1   crapule2  crapule3
# 3emes tests (95%) : inconnuA inconnuB inconnuC

## RECUPERATION IMAGE CORROMPUE
# récuperation/lecture de l'image corrompue
# ne pas modifier
image_defect= plt.imread("corrompue-chien.png", True)
# visualisation de cette image
plt.figure(0)
plt.imshow(image_defect,cmap="gray")
plt.show()

# %%
mask=np.ones_like(image_defect)
mask[image_defect==0]=0  
mask

# %%
def matrice_A(b, beta, beta_prec, Mask, omega):
    # Assurer que toutes les matrices ont la même forme
    assert b.shape == beta.shape == beta_prec.shape == Mask.shape, "All matrices must have the same dimensions"

    # Utiliser n et m pour des matrices potentiellement non carrées
    n, m = b.shape
    
    # Initialiser A comme une matrice n x m de zéros
    A = np.zeros((n, m))
    
    # Remplissage de la matrice A
    for i in range(n):
        for j in range(m):
            if Mask[i, j]:  # Supposons que Mask soit une matrice booléenne indiquant les zones à traiter
                # Calculer la différence pondérée entre beta et beta_prec ajustée par le facteur omega
                A[i, j] = omega * (beta[i, j] * b[i, j] - beta_prec[i, j] * b[i, j])
            else:
                # Si le masque est False, on pourrait laisser cette valeur à 0 ou définir une autre valeur par défaut
                A[i, j] = 0  # Cette valeur peut être ajustée si nécessaire

    return A

# %%
def getImage(b, beta, mask):
    # Créer une copie de beta pour éviter de modifier l'original lors de la reconstruction
    Im = beta.copy()

    # Parcourir chaque élément du masque
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # Si le masque à cette position est False, cela peut signifier que la valeur de beta
            # devrait être utilisée telle quelle. S'il est True, cela peut indiquer une zone corrompue
            # ou spéciale où nous pourrions vouloir combiner beta et b d'une certaine manière.
            if mask[i, j]:
                # Exemple d'ajustement : moyenne pondérée de beta et b à cette position
                # Le facteur de pondération peut être ajusté selon les besoins spécifiques
                Im[i, j] = 0.5 * beta[i, j] + 0.5 * b[i, j]
    
    return Im


# %%
def getCritere(beta_prec,beta):
    return np.linalg.norm(beta_prec-beta)

# %%
b = image_defect  # Assurez-vous que cette variable est définie correctement
beta = b.copy()
eps = 1.e-3
iter = 0
omega = 0.01  # paramètre de relaxation
N = 40  # nb d'itération max
mask = np.ones_like(b, dtype=bool)  # Définissez correctement

n, m = b.shape  # Taille de l'image

for iter in range(N):
    # Parcourir chaque pixel
    for i in range(n):
        for j in range(m):
            if mask[i, j]:  # Si le pixel est dans la zone à débruiter
                # Calculer la somme des contributions des pixels voisins
                somme = 0
                for ii in range(max(0, i - 1), min(n, i + 2)):
                    for jj in range(max(0, j - 1), min(m, j + 2)):
                        if mask[ii, jj]:  # Si le pixel voisin est dans la zone à débruiter
                            somme += beta[ii, jj] * b[ii, jj]

                # Calculer la nouvelle valeur de beta pour ce pixel
                beta_new = (omega / (1 + 4 * omega)) * (somme - b[i, j]) + (1 - omega) * beta[i, j]

                # Mettre à jour la valeur de beta
                beta[i, j] = beta_new

# Calculer le critère d'arrêt
critere = getCritere(beta_new, beta)
beta_prec = beta.copy()  # Sauvegarder l'ancienne estimation

# Renvoyer l'image dé bruitée
beta_denoised = beta.copy()
plt.imshow(beta_denoised, cmap='gray')
plt.show()

# %%


