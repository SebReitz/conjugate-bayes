import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import gamma, norm, invgamma, expon, poisson
from scipy.special import beta as beta_fc
from scipy.special import gamma as gamma_fc 


"""
Parent class
"""
class Probability_Distribution:
    def __init__(self, name):
        self.name = name
        self.pdf = None
        self.variance = None
        self.parameters = None

    def print_stuff(self):
        print("TEST")

    def std(self):
        std = math.sqrt(self.variance())
        return std

    def plot_pdf(self, x0, xn, pmf=False, color="blue"):     
        if not pmf:
            domain = np.linspace(x0, xn, 1000)
            y_pdf = [self.pdf(x) for x in domain]
            return plt.plot(domain, y_pdf, color=color)
        else:
            domain = np.linspace(x0, xn, xn-x0+1, dtype=int)
            y_pdf = [self.pdf(x) for x in domain]
            return plt.scatter(domain, y_pdf, color=color)

    def plot_cdf(self, x0, xn, pmf=False):
        if not pmf:
            domain = np.linspace(x0, xn, 1000)
            y_cdf = [self.cdf(x) for x in domain]
            return plt.plot(domain, y_cdf)
        else:
            domain = np.linspace(x0, xn, xn-x0+1, dtype=int)
            y_cdf = [self.cdf(x) for x in domain]
            return plt.scatter(domain, y_cdf)

    def describe(self):
        return {"name": self.name, "parameters": self.parameters}


"""
Child classes: Continuous Likelihood
"""
class Normal(Probability_Distribution):
    """Initialise Normal distribution."""
    def __init__(self, mu, sigma2, tau=False):
        self.name = "Normal"
        self.mu = mu
        self.sigma2 = sigma2 if not tau else 1/tau
        self.parameters = {"mu": self.mu, "sigma2": self.sigma2}

    def pdf(self, X):
        pdf = 1/math.sqrt(self.sigma2*math.pi*2)*math.exp(
            -0.5*((X-self.mu)/self.sigma2)**2)
        return pdf

    def cdf(self, X):
        cdf = norm.cdf(X, loc=self.mu, scale=self.sigma)
        return cdf

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
        self.parameters = {"alpha": self.alpha, "beta": self.beta}

    def gamma_fct(self, a):
        return gamma_fc(a)

    def pdf(self, X):
        pdf = (self.beta**self.alpha)/self.gamma_fct(self.alpha)\
        *X**(self.alpha-1)*math.exp(-self.beta*X)
        return pdf

    def cdf(self, X):
        cdf = gamma.cdf(X, self.alpha, scale=1/self.beta)
        return cdf

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
        self.parameters = {"alpha": self.alpha, "beta": self.beta}

    def gamma_fct(self, a):
        return gamma_fc(a)

    def pdf(self, X):
        pdf = (self.beta**self.alpha)/self.gamma_fct(self.alpha)\
        *X**(-self.alpha-1)*math.exp(-self.beta/X)
        return pdf

    def cdf(self, X):
        cdf = invgamma.cdf(X, self.alpha, scale=self.beta)
        return cdf

    def mean(self):
        mean = self.beta/(self.alpha-1) if self.alpha > 1 else 9999
        return mean

    def variance(self):
        var = self.beta**2/((self.alpha-1)**2*(self.alpha-2)) if \
        self.alpha > 2 else 9999
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


class Exponential(Probability_Distribution):
    """Initialise Exponential distribution."""
    def __init__(self, lambda_):
        self.name = "Exponential"
        self.lambda_ = lambda_
        self.parameters = {"lambda": self.lambda_}

    def pdf(self, X):
        pdf = self.lambda_*math.exp(-self.lambda_*X) if X >=0 else 0
        return pdf

    def cdf(self, X):
        pdf = 1-math.exp(-self.lambda_*X) if X >=0 else 0
        return pdf

    def mean(self):
        mean = 1/self.lambda_
        return mean

    def variance(self):
        var = 1/self.lambda_**2
        return var

    def equitailed_cs(self, alpha2):
        """
        Calculates the equitailed credible set of a parameter.
        The alpha to be inserted should be between (0-100).
        """
        alpha_split = (100-alpha2)/200
        lower_bound = expon.ppf(alpha_split, scale =1/self.lambda_)
        upper_bound = expon.ppf(1-alpha_split, scale =1/self.lambda_)
        return (lower_bound, upper_bound)


"""
Child classes: Discrete Likelihood
"""
class Bernoulli(Probability_Distribution):
    """Initialise Bernoulli distribution."""
    def __init__(self, p):
        self.name = "Bernoulli"
        self.p = p
        self.parameters = {"probability": self.p}


class Beta(Probability_Distribution):
    """Initialise Beta distribution."""
    def __init__(self, alpha, beta):
        self.name = "Beta"
        self.alpha = alpha
        self.beta = beta
        self.parameters = {"alpha": self.alpha, "beta": self.beta}

    def pdf(self, X):
        pmf = (X**(self.alpha-1)*(1-X)**(self.beta-1))/\
        beta_fc(self.alpha, self.beta)
        return pmf


class Poisson(Probability_Distribution):
    """Initialise Poisson distribution."""
    def __init__(self, lambda_):
        self.name = "Poisson"
        self.lambda_ = lambda_
        self.parameters = {"lambda": self.lambda_}

    def pdf(self, X):
        pmf = poisson.pmf(X, self.lambda_)
        return pmf

    def cdf(self, X):
        cdf = poisson.cdf(X, self.lambda_)
        return cdf

    def mean(self):
        mean = self.lambda_
        return mean

    def variance(self):
        var = self.lambda_
        return var

