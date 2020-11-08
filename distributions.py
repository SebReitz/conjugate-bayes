import numpy as np
import matplotlib.pyplot as plt
import math


class Probability_Distribution:
    def __init__(self, name):
        self.name = name
        self.pdf = None

    def print_stuff(self):
        print("TEST")

    def plot_pdf(self, x0, xn):
        domain = np.linspace(x0, xn, 1000)
        y_pdf = [self.pdf(x) for x in domain]
        return plt.plot(domain, y_pdf)



class Normal(Probability_Distribution):
    """Initialise Normal distribution."""
    def __init__(self, mu, sigma):
        self.name = "Normal"
        self.mu = mu
        self.sigma = sigma

    def pdf(self, X):
        pdf = 1/math.sqrt(self.sigma*math.pi*2)*math.exp(
            -0.5*((X-self.mu)/self.sigma)**2)
        return pdf


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

































