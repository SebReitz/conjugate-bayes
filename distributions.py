import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gamma, norm, invgamma


class Probability_Distribution:
    def __init__(self, name):
        self.name = name
        self.pdf = None
        self.variance = None

    def print_stuff(self):
        print("TEST")

    def std(self):
        std = math.sqrt(self.variance())
        return std

    def plot_pdf(self, x0, xn):
        domain = np.linspace(x0, xn, 1000)
        y_pdf = [self.pdf(x) for x in domain]
        return plt.plot(domain, y_pdf)


class Normal(Probability_Distribution):
    """Initialise Normal distribution."""
    def __init__(self, mu, sigma2):
        self.name = "Normal"
        self.mu = mu
        self.sigma = sigma2

    def pdf(self, X):
        pdf = 1/math.sqrt(self.sigma2*math.pi*2)*math.exp(
            -0.5*((X-self.mu)/self.sigma2)**2)
        return pdf

    def mean(self):
        mean = self.mu
        return mean

    def variance(self):
        var = self.sigma2
        return var
    
    def precision(self):
        precision = 1/self.sigma2
        return precision

    def equitailed_cs(self, alpha2):
        """
        Calculates the equitailed credible set of a parameter.
        The alpha to be inserted should be between (0-100).
        """
        alpha_split = (100-alpha2)/200
        lower_bound = norm.ppf(alpha_split, loc=self.mu, scale=self.sigma)
        upper_bound = norm.ppf(1-alpha_split, loc=self.mu, scale=self.sigma)
        return (lower_bound, upper_bound)


class Gamma(Probability_Distribution):
    """Initialise Gamma distribution."""
    def __init__(self, alpha, beta):
        self.name = "Gamma"
        self.alpha = alpha
        self.beta = beta

    def gamma_fct(self, a):
        return math.factorial(a-1)

    def pdf(self, X):
        pdf = (self.beta**self.alpha)/self.gamma_fct(self.alpha)*X**(self.alpha-1)*math.exp(
            -self.beta*X)
        return pdf

    def mean(self):
        mean = self.alpha/self.beta
        return mean

    def variance(self):
        var = self.alpha/self.beta**2
        return var

    def equitailed_cs(self, alpha2):
        """
        Calculates the equitailed credible set of a parameter.
        The alpha to be inserted should be between (0-100).
        """
        alpha_split = (100-alpha2)/200
        lower_bound = gamma.ppf(alpha_split, self.alpha, scale=1/self.beta)
        upper_bound = gamma.ppf(1-alpha_split, self.alpha, scale=1/self.beta)
        return (lower_bound, upper_bound)


class Inv_Gamma(Probability_Distribution):
    """Initialise Inverse Gamma distribution."""
    def __init__(self, alpha, beta):
        self.name = "Inv_Gamma"
        self.alpha = alpha
        self.beta = beta

    def gamma_fct(self, a):
        return math.gamma(a)

    def pdf(self, X):
        pdf = (self.beta**self.alpha)/self.gamma_fct(self.alpha)*X**(-self.alpha-1)*math.exp(
            -self.beta/X)
        return pdf

    def mean(self):
        mean = self.beta/(self.alpha-1) if self.alpha > 1 else 9999
        return mean

    def variance(self):
        var = self.beta**2/((self.alpha-1)**2*(self.alpha-2)) if self.alpha > 2 else 9999
        return var

    def equitailed_cs(self, alpha2):
        """
        Calculates the equitailed credible set of a parameter.
        The alpha to be inserted should be between (0-100).
        """
        alpha_split = (100-alpha2)/200
        lower_bound = invgamma.ppf(alpha_split, self.alpha, scale=self.beta)
        upper_bound = invgamma.ppf(1-alpha_split, self.alpha, scale=self.beta)
        return (lower_bound, upper_bound)


























