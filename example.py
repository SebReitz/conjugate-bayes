from conjugates import get_conjugate, CONJUGATES_INV
from distributions import Normal, Gamma


# Our base model is a normal distribution, with the likelihood
likelihood = Normal(1, [])  # with mean 1 and unknown precision tau = 1/variance
# We observe the values for X to be [1,2,1,0.5]
observations = [1,2,1,0.5,2,4,-1,-5,2,4,5]
# We want to find out what conjugate priors there are for a Normal distribution
# and its precision tau

print(CONJUGATES_INV[("Normal", "tau")])
# The answer is a Gamma distribution. Let's set the prior to Gamma:
prior = Gamma(3, 1)  # This is not an uninformative prior

# Combining these to the posterior:
posterior = get_conjugate(likelihood, prior, "precision", observations)
print(posterior.describe())
print(posterior.mean())
print(posterior.equitailed_cs(95))
posterior.plot_pdf(0, 2)

