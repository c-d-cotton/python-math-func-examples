#!/usr/bin/env python3
# PYTHON_PREAMBLE_START_STANDARD:{{{

# Christopher David Cotton (c)
# http://www.cdcotton.com

# modules needed for preamble
import importlib
import os
from pathlib import Path
import sys

# Get full real filename
__fullrealfile__ = os.path.abspath(__file__)

# Function to get git directory containing this file
def getprojectdir(filename):
    curlevel = filename
    while curlevel is not '/':
        curlevel = os.path.dirname(curlevel)
        if os.path.exists(curlevel + '/.git/'):
            return(curlevel + '/')
    return(None)

# Directory of project
__projectdir__ = Path(getprojectdir(__fullrealfile__))

# Function to call functions from files by their absolute path.
# Imports modules if they've not already been imported
# First argument is filename, second is function name, third is dictionary containing loaded modules.
modulesdict = {}
def importattr(modulefilename, func, modulesdict = modulesdict):
    # get modulefilename as string to prevent problems in <= python3.5 with pathlib -> os
    modulefilename = str(modulefilename)
    # if function in this file
    if modulefilename == __fullrealfile__:
        return(eval(func))
    else:
        # add file to moduledict if not there already
        if modulefilename not in modulesdict:
            # check filename exists
            if not os.path.isfile(modulefilename):
                raise Exception('Module not exists: ' + modulefilename + '. Function: ' + func + '. Filename called from: ' + __fullrealfile__ + '.')
            # add directory to path
            sys.path.append(os.path.dirname(modulefilename))
            # actually add module to moduledict
            modulesdict[modulefilename] = importlib.import_module(''.join(os.path.basename(modulefilename).split('.')[: -1]))

        # get the actual function from the file and return it
        return(getattr(modulesdict[modulefilename], func))

# PYTHON_PREAMBLE_END:}}}

import functools
import numpy as np
from scipy.stats import norm

# Basic Functions:{{{1
def mh_basic():
    """
    Take 100 draws from an N(mean, std) distribution
    """

    mean = 0.1
    std = 0.3

    # get random draws from N(mean, std) distribution
    draws = np.random.normal(loc = mean, scale = std, size = 1000)

    # compute posterior function based upon draws
    def loglikelihoodfunc(values):
        mean = values[0]
        # std = values[1]
        density_separate = norm.pdf(draws, loc = mean, scale = 0.3)
        ll = np.sum(np.log(density_separate))
        return(ll)

    results = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'metropolis_hastings')(loglikelihoodfunc, [0.1], [0], 10000, printdetails = False, logposterior = True)
    print(np.mean(np.array(results)[1000: , :], axis = 0))


def mh_save():
    """
    Save the data and then load it
    """

    mean = 0.1
    std = 0.3

    # get random draws from N(mean, std) distribution
    draws = np.random.normal(loc = mean, scale = std, size = 1000)

    # compute posterior function based upon draws
    def loglikelihoodfunc(values):
        mean = values[0]
        # std = values[1]
        density_separate = norm.pdf(draws, loc = mean, scale = 0.3)
        ll = np.sum(np.log(density_separate))
        return(ll)

    # save results
    importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'metropolis_hastings')(loglikelihoodfunc, [0.1], [0], 10000, printdetails = False, logposterior = True, savefile = __projectdir__ / Path('bayes/temp/mh_save/results.csv'))

    # load results
    results = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'distfromfile')(__projectdir__ / Path('bayes/temp/mh_save/results.csv'), burnindelete = 1000)
    print(np.mean(results, axis = 0))
    

# Pooling:{{{1
def mh_pool_aux_loglikelihoodfunc(mean, draws, values):
    """
    Auxilliary function for mh_pool
    Generates the likelihood given all necessary parameters
    """
    mean = values[0]
    # std = values[1]
    density_separate = norm.pdf(draws, loc = mean, scale = 0.3)
    ll = np.sum(np.log(density_separate))
    return(ll)


def mh_pool():
    """
    Run the Bayesian estimation in different processors and combine at the end.
    """
    mean = 0.1
    std = 0.3

    # get random draws from N(mean, std) distribution
    draws = np.random.normal(loc = mean, scale = std, size = 100)

    # compute posterior function based upon draws
    loglikelihoodfunc = functools.partial(mh_pool_aux_loglikelihoodfunc, mean, draws)

    # save results
    importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'metropolis_hastings_pool')(__projectdir__ / Path('bayes/temp/mh_pool/'), loglikelihoodfunc, [0.1], [0.1], 1000, printdetails = False, logposterior = True)

    # load results
    results = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'distfromfolder')(__projectdir__ / Path('bayes/temp/mh_pool/'), burnindelete = 100, concatenate = True)
    print(np.mean(results, axis = 0))


# Bounds/Priors:{{{1
def mh_bounds():
    """
    Add a lower bound so I can consider standard deviations
    Note that when we consider standard deviations, we need a lower bound otherwise the likelihood function fails (we could also add a prior limiting the sample to only positive values)
    Note that I also avoid making the scaling parameter too large for the standard deviation
    """

    mean = 0.1
    std = 0.3

    # get random draws from N(mean, std) distribution
    draws = np.random.normal(loc = mean, scale = std, size = 1000)

    # compute posterior function based upon draws
    def loglikelihoodfunc(values):
        mean = values[0]
        std = values[1]
        density_separate = norm.pdf(draws, loc = mean, scale = std)
        ll = np.sum(np.log(density_separate))
        return(ll)

    results = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'metropolis_hastings')(loglikelihoodfunc, [0.1, 0.1], [0, 0.1], 10000, lowerboundlist = [None, 0], printdetails = False, logposterior = True)
    print(np.mean(np.array(results)[1000: , :], axis = 0))


def mh_priors():
    """
    Add my standard method for getting priors
    """

    np.random.seed(42)

    mean = 0.1
    std = 0.3

    # get random draws from N(mean, std) distribution
    draws = np.random.normal(loc = mean, scale = std, size = 1000)

    # compute basic log-likelihood function based upon draws
    def loglikelihoodfunc(values):
        mean = values[0]
        std = values[1]
        density_separate = norm.pdf(draws, loc = mean, scale = std)
        ll = np.sum(np.log(density_separate))
        return(ll)

    # get priors
    priorlist_meansd = [['gamma', 0.15, 0.05], ['invgamma', 0.3, 0.05]]
    # get a priorlist based upon parameters rather than means and standard deviations
    priorlist_parameters = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'getpriorlist_convert')(priorlist_meansd)
    # get details on priors
    prior_means, prior_sds, prior_lbs, prior_ubs = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'getpriorlistdetails_parameters')(priorlist_parameters)
    # get a density function for the priors - this is NOT log
    priorfunc = functools.partial(importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'getpriordensityfunc_aux'), priorlist_parameters)
    # get scalelist based upon sds
    scalelist = [0.5 * sd for sd in prior_sds]

    def posteriorfunc(values):
        """
        Get the posterior function incorporating both the priors and the log-likelihood
        Note that we log the prior function
        """
        return(np.log(priorfunc(values)) + loglikelihoodfunc(values))

    results = importattr(__projectdir__ / Path('submodules/python-math-func/bayesian_func.py'), 'metropolis_hastings')(posteriorfunc, scalelist, prior_means, 10000, lowerboundlist = prior_lbs, upperboundlist = prior_ubs, printdetails = False, logposterior = True)
    print(np.mean(np.array(results)[1000: , :], axis = 0))

