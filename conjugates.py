from distributions import Normal, Gamma, Inv_Gamma


CONJUGATE_TABLE = {
              ("Normal", "Normal", "mean"): "normal_normal_mean",
              ("Normal", "Gamma", "precision"): "normal_gamma_precision",
              ("Normal", "Inv_Gamma", "variance"): "normal_inv_gamma_variance",
              ("Exponential", "Gamma", "rate"): "exponential_gamma_rate",
              ("Gamma", "Gamma", "rate"): "gamma_gamma_rate",
              ("Poisson", "Gamma", "rate"): "poisson_gamma_rate"
              }

CONJUGATES = {("Normal", "Normal"): {"Posterior": "Normal", "On": "mu"},
              ("Normal", "Gamma"): {"Posterior": "Gamma", "On": "tau"},
              ("Normal", "Inv_Gamma"): {"Posterior": "Inv_Gamma", "On": "sigma2"},
              ("Exponential", "Gamma"): {"Posterior": "Gamma", "On": "lambda_"},
              ("Gamma", "Gamma"): {"Posterior": "Gamma", "On": "beta"},
              ("Poisson", "Gamma"): {"Posterior": "Gamma", "On": "lambda_"}
              }

CONJUGATES_INV = {("Normal", "mu"): "Normal",
              ("Normal", "tau"): "Gamma",
              ("Normal", "sigma2"): "Inv_Gamma",
              ("Exponential", "lambda_"): "Gamma",
              ("Gamma", "beta"): "Gamma",
              ("Poisson", "lambda_"): "Gamma"
              }

assert len(CONJUGATE_TABLE)==len(CONJUGATES)==len(CONJUGATES_INV)


def get_conjugate(likelihood, prior, parameter: str, observations: list):
    conjugate = CONJUGATE_TABLE[(likelihood.name, prior.name, parameter)]

    n = len(observations)

    if conjugate == "normal_normal_mean":
        new_mean = (1/(1/prior.sigma2 + n/likelihood.sigma2))*\
                    (prior.mu/prior.sigma2+sum(observations)/likelihood.sigma2)
        new_var = 1/(1/prior.sigma2+n/likelihood.sigma2)
        posterior = Normal(new_mean, new_var)

    elif conjugate == "normal_gamma_precision":
        new_alpha = prior.alpha + n/2
        new_beta = prior.beta + sum([(observations[i]-likelihood.mu)**2
                                for i in range(n)])/2
        posterior = Gamma(new_alpha, new_beta)

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



