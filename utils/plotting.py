from scipy.stats import gaussian_kde
def plot_pnl_distribution(l , title):
    plt.figure(figsize=(6, 4))
    for (pnl_vector , label , color, style) in l:
        density = gaussian_kde(pnl_vector)
        x = np.linspace(pnl_vector.min(), pnl_vector.max(), 1000)
        plt.plot(x, density(x), label=label, color=color , linestyle=style)
        
    plt.title(f'P&L Distributions for {title} Option')
    #plt.xlim(-0.04, 0.04)
    plt.xlabel('P&L')
    plt.ylabel('Density')
    plt.legend()
    plt.grid()
    plt.show()


def plot_sigma_comparison(sigma_true, sigma_pred):
    plt.figure(figsize=(10,4))
    for i in range(5):
        plt.plot(sigma_true[i], alpha=0.5, color='blue')
        plt.plot(sigma_pred[i], alpha=0.5, color='red', linestyle='--')
    plt.title('Sigma vrai (bleu) vs Sigma prédit (rouge)')
    plt.show()    
