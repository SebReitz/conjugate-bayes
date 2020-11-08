import numpy as np

from distributions import Normal


CONJUGATE_TABLE = {("Normal", "Normal", "mu"): "Normal_Normal_mu"
    
    
    
    
    }





def get_conjugate(likelihood, prior, parameter: str, observations: list):
    lk_kind = likelihood.name
    prior_kind = prior.name
    conjugate = CONJUGATE_TABLE[(lk_kind, prior_kind, parameter)]
    
    n = len(observations)
    
    if conjugate == "Normal_Normal_mu":
         posterior = Normal(prior.sigma*np.sum(observations)/(likelihood.sigma/n+prior.sigma)
                            + likelihood.sigma*prior.mu/(likelihood.sigma/n+prior.sigma),
                            1/((1/prior.sigma)+(n/likelihood.sigma)))
    return posterior

































