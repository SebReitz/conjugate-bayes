import numpy as np

from distributions import Normal, Gamma, Inv_Gamma


CONJUGATE_TABLE = {
              ("Normal", "Normal", "mean"): "normal_normal_mean",
              ("Normal", "Gamma", "precision"): "normal_gamma_precision",
              ("Normal", "Inv_Gamma", "variance"): "normal_inv_gamma_variance",
              ("Exponential", "Gamma", "rate"): "exponential_gamma_rate",
              ("Gamma", "Gamma", "rate"): "gamma_gamma_rate",
              ("Poisson", "Gamma", "rate"): "poisson_gamma_rate"
              }

CONJUGATES = {("Normal", "Normal"): {"Posterior": "Normal", "On": "Mu"},
              ("Normal", "Gamma"): {"Posterior": "Gamma", "On": "Tau"},
              ("Normal", "Inv_Gamma"): {"Posterior": "Inv_Gamma", "On": "Sigma2"},
              ("Exponential", "Gamma"): {"Posterior": "Gamma", "On": "Lambda"},
              ("Gamma", "Gamma"): {"Posterior": "Gamma", "On": "Beta"},
              ("Poisson", "Gamma"): {"Posterior": "Gamma", "On": "Lambda"}
              }

CONJUGATES_INV = {("Normal", "Mu"): "Normal",
              ("Normal", "Tau"): "Gamma",
              ("Normal", "Sigma2"): "Inv_Gamma",
              ("Exponential", "Lambda"): "Gamma",
              ("Gamma", "Beta"): "Gamma",
              ("Poisson", "Lambda"): "Gamma"
              }


def get_conjugate(likelihood, prior, parameter: str, observations: list):
    conjugate = CONJUGATE_TABLE[(likelihood.name, prior.name, parameter)]

    n = len(observations)

    if conjugate == "normal_normal_mean":
        posterior = Normal(prior.sigma*np.sum(observations)/
                            (likelihood.sigma/n+prior.sigma)
                            + likelihood.sigma*prior.mu/
                            (likelihood.sigma/n+prior.sigma),
                            1/((1/prior.sigma)+(n/likelihood.sigma)))

    elif conjugate == "normal_gamma_precision":
        posterior = Gamma(prior.alpha + n/2, prior.beta +
                           sum([(observations[i]-likelihood.mu)**2
                                for i in range(n)])/2)

    elif conjugate == "normal_inv_gamma_variance":
        posterior = Inv_Gamma(prior.alpha + n/2, prior.beta +
                           sum([(observations[i]-likelihood.mu)**2
                                for i in range(n)])/2)

    elif conjugate == "exponential_gamma_rate":
        posterior = Gamma(prior.alpha+n, prior.beta+sum(observations))

    elif conjugate == "gamma_gamma_rate":
        posterior = Gamma(prior.alpha+n*likelihood.alpha, prior.beta+sum(observations))

    elif conjugate == "poisson_gamma_rate":
        posterior = Gamma(prior.alpha+sum(observations), prior.beta+n)

    return posterior

































