import numpy as np
def simulate_heston(S0 , V0, kappa , theta , eta , rho , T , J , I, seed):
    rng = np.random.default_rng(seed)
    #on simule deux gaussiennes standards et une autre pour avoir une  correlation rho
    Z1=rng.standard_normal((I , J))
    Z2=rng.standard_normal((I , J))
    Z0= rho * Z1 + np.sqrt(1- rho**2) *Z2
    #on coupe l'horizon du temps T  sur J+1 pas de temps
    time_grid = np.linspace(0, T,J+1)
    S=np.zeros((I , J+1))
    S[:,0]=S0
    V=np.zeros((I, J+1))
    V[:, 0] = V0
    sigma= np.zeros((I , J+1))
    sigma[:,0]=np.sqrt(V0)
    dt=T/J
    for j in range(J):
        # 1/la volatilité (ou la avriance ) ne doit pas etre negative
        # 2/ on va utilisé une racine et le bruit engendré par le mouvement brownien peut emmener V a etre negative donc on utilise V_pos
        V_pos=np.maximum(V[:,j] , 0)
        V[: ,j+1]= V_pos +kappa*(theta-V_pos)*dt+eta * np.sqrt(V_pos)*np.sqrt(dt)*Z1[:,j]
        S[:, j+1]=S[: , j]* np.exp(-1/2 * V_pos * dt + np.sqrt(V_pos) * np.sqrt(dt) *Z0[:,j])
        sigma[:, j+1] = np.sqrt(np.maximum(V[:, j+1], 0.0))
    # on a besoin de la diffeernce entre les prix a chaque pas du temps pour calculer la fonction objective
    dS=np.diff(S, axis=1)
    t_expanded = np.tile(time_grid, (I, 1))
    #states est la variable d'entrée pour chaque reseau de neuronnes , contient t,S, sigma
    states = np.stack([t_expanded, S, sigma], axis=-1)

    return S , V , dS , sigma , states

def simulate_shifted_bergomi(S0 , varsigma, f0,epsilon ,H, rho , T , J , I, seed):
    rng = np.random.default_rng(seed)
    #on simule deux gaussiennes standards et une autre pour avoir une  correlation rho
    Z1=rng.standard_normal((I , J))
    Z2=rng.standard_normal((I , J))
    Z0= rho * Z1 + np.sqrt(1- rho**2) *Z2
    #on coupe l'horizon du temps T  sur J+1 pas de temps
    time_grid = np.linspace(0, T,J+1)
    S=np.zeros((I , J+1))
    S[:,0]=S0
    sigma= np.zeros((I , J+1))
    sigma[:,0]=np.sqrt(f0)
    dt = T / J
    dW = Z1 * np.sqrt(dt)  # incréments browniens shape (I, J)
    W = np.cumsum(dW, axis=1)  # shape (I, J)
    # Ajoute W_0 = 0 au début pour representer W_0
    W = np.concatenate([np.zeros((I, 1)), W], axis=1)  # shape (I, J+1)

    for j in range(J):
        # Noyau pour le pas j : vecteur de taille j
        if j > 0:
            k_indices = np.arange(j)
            kernel = (epsilon + time_grid[j] - time_grid[k_indices])**(H - 0.5)
            # Intégrale : somme sur k=0..j-1, shape (I,)
            integral = np.sum(kernel[None, :] * dW[:, :j], axis=1)
        else:
            integral = np.zeros(I)
        
        # Volatilité à t_j
        correction = varsigma**2 / (8*H) * ((epsilon + time_grid[j])**(2*H) - epsilon**(2*H))
        sigma_j = np.sqrt(f0) * np.exp(varsigma/2 * integral - correction)
        
        # Prix
        S[:, j+1] = S[:, j] * np.exp(-0.5 * sigma_j**2 * dt + sigma_j * np.sqrt(dt) * Z0[:, j])
        sigma[:, j+1] = sigma_j 
    # on a besoin de la diffeernce entre les prix a chaque pas du temps pour calculer la fonction objective
    dS=np.diff(S, axis=1)
    # une matrice de taille (I , J+1) ou chaque ligne est la grille de temps pour chaque trajectoire , pour pouvoir concatener avec S et sigma et avoir une matrice d'entrée de taille (I , J+1 , 3) pour les reseaux de neuronnes
    t_expanded = np.tile(time_grid, (I, 1))
    #states est la variable d'entrée pour chaque reseau de neuronnes , contient t,S, sigma
    states = np.stack([t_expanded, S, sigma], axis=-1)
    W_hat = np.stack([t_expanded, W], axis=-1)  # shape (I, J+1, 2)

    return S , dS , sigma , states , W_hat
