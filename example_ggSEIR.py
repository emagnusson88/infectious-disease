import pandas as pd
import numpy as np
import math
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.stats import poisson
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot


def ggSEIR(y, t, params, conds, flag):
    '''
    This function models SEIR model with time-dependent transmission rate
    

    Parameters
    ----------
    y : array_like with shape (n,) 
        incidence vector
    t : array_like with shape (n,) 
        time vector
    params : array_like with shape (n,) or float
        initial guess of parameters to be optimized
    conds : array_like with shape (n,) or float
        values of fixed parameters in the model
    flag : int
        indicate function of transmission rate
        1 - exponential decline
        2 - hyperbolic decline
        3 - harmonic decline
        4 - constant
        

    Returns
    -------
    None.

    '''
#time-dependent beta(t): exponential decline

#fixed conditions
    N=conds[0]
    k= conds[1]
    gam=conds[2]

#parameters
    beta=params[0]

    if flag==1: #exponential decline
        q=params[1]
        phi=gam/beta0
        beta=beta0*((1-phi)*math.exp(-q*t) + phi)
    elif flag==2: #hyperbolic decline
        q=params[1]
        v=params[2]
        phi=gam/beta0
        beta=beta0*((1-phi)*(1/(1+q*v*t)**(1/v)) + phi)
    elif flag==3: #harmonic decline
        q=params[1]
        phi=gam/beta0
        beta=beta0*((1-phi)*(1/(1+q*t)) + phi)
    elif flag==4:
        beta=beta0 #original SEIR model

#differential equations
    S, E, I, R, C = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - k * E
    dIdt = k*E - gam * I
    dRdt = gam * I
    dCdt = k*E #cumulative incidence of infectious

    return dSdt, dEdt, dIdt, dRdt, dCdt


# Integrate the SIR equations over the time grid, t.
def solved_ggSEIR(params, y, t, conds, flag):
    '''
    This function solves the system of differential equations in ggSEIR

    Parameters
    ----------
    y : array_like with shape (n,) 
        incidence vector
    t : array_like with shape (n,) 
        time vector
    params : array_like with shape (n,) or float
        initial guess of parameters to be optimized
    conds : array_like with shape (n,) or float
        values of fixed parameters in the model
    flag : int
        indicate function of transmission rate
        1 - exponential decline
        2 - hyperbolic decline
        3 - harmonic decline
        4 - constant
        

    Returns
    -------
    S : TYPE
        DESCRIPTION.
    E : TYPE
        DESCRIPTION.
    I : TYPE
        DESCRIPTION.
    R : TYPE
        DESCRIPTION.
    cum_incidence : TYPE
        DESCRIPTION.

    '''
    ret = odeint(ggSEIR, y, t, args=(params, conds, flag))
    S, E, I, R, cum_incidence = ret.T    
    return S, E, I, R, cum_incidence

def predicted(params, IC, tf,  conds, flag):
    '''
    predict incidence over a time period

    Parameters
    ----------
    params : array_like with shape (n,) or float
        initial guess of parameters to be optimized
    IC : array_like with shape (5,)
        initial condition of the differential equation system,
        providing S0, E0, I0, R0, and C0
    tf : array_like with shape (n,) 
        time vector
    conds : rray_like with shape (n,) or float
        values of fixed parameters in the model
    flag : Tint
        indicate function of transmission rate
        1 - exponential decline
        2 - hyperbolic decline
        3 - harmonic decline
        4 - constant
        
    Returns
    -------
    incidence : array_like with shape (n,) 
        estimated incidence vector

    '''
    S, E, I, R, cum_incidence= solved_ggSEIR(params, IC, tf, conds, flag)
    incidence = pd.Series(cum_incidence).diff()
    incidence.iat[0] = cum_incidence[0]
    return incidence

def ResidFun(params, IC, tf,  conds, flag, cases):
    '''
    calculate residual = predicted - observed

    Parameters
    ----------
    params : array_like with shape (n,) or float
        initial guess of parameters to be optimized
    IC : array_like with shape (5,)
        initial condition of the differential equation system,
        providing S0, E0, I0, R0, and C0
    tf : array_like with shape (n,) 
        time vector
    conds : rray_like with shape (n,) or float
        values of fixed parameters in the model
    flag : Tint
        indicate function of transmission rate
        1 - exponential decline
        2 - hyperbolic decline
        3 - harmonic decline
        4 - constant

    Returns
    -------
    resid : array_like with shape (n,) 
        residual = predicted - observed

    '''
    incidence = predicted(params, IC, tf,  conds, flag)
    z = incidence.sub(cases, axis=0)
    resid = (z.values).ravel()
    return resid


def fitting_ggSEIR(cases, IC, inits, bounds, timevect, conds, flag, method='trf'):
    '''
    cases: vector of incidence
    timevect: vector of time points
    method: least_squares fitting method
    '''
    P = least_squares(ResidFun, inits, bounds=bounds, method=method, ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000000, args=(IC, timevect, conds, flag, cases))
    Ptrue = P["x"]
    return Ptrue


def confint(Ptrue, bestfit, cases, IC, inits, bounds, timevect, conds, flag, nsim, dist, factor=None, method ='trf'):
    '''
    Generate distribution of paramaters by running nsim simulations
    
    Parameters
    ----------
    Ptrue: solution from fittingSimpleGrowth
    bestfit: predicted incidence from model
    cases: actual case
    inits: inital guess of parameters
    bounds: lower and upper bounds of parameters
    timevect: time vector
    flag: type of model to fit
    method: method of least_squares function, default 'trf', specify method='lm' if Levenberg-Marquardt
    nsim: number of simulations to run
    dist: distribution assumed for error structure (Poisson, Negative Binomial, or Lognormal)
    *args: if dist==2, give factor = xxx
    '''
    Phats = pd.DataFrame(np.zeros((nsim, len(Ptrue))))
    resid = bestfit.sub(cases, axis=0)
    variance = np.var(resid)
    curves = pd.DataFrame([])
    
    
    for simulation in range(nsim):
        yirData = pd.Series(np.zeros((len(bestfit))))
        yirData.iat[0] = bestfit.iat[0,0]
        
        if dist==1: #Poisson error structure
            for t in range(1, len(bestfit), 1):
                yirData.iloc[t] = poisson.rvs(bestfit.iat[t,0], 1)
        elif dist==2: # Negative Binomial error structure
            for t in range(1, len(bestfit), 1):
                mean = bestfit.iat[t,0]
                var = mean * factor
                param2 = mean/var
                param1 = mean * param2/(1-param2)
                yirData.iloc[t] = nbinom.rvs(param1, param2, mean)
        elif dist==3:
            for t in range(1, len(bestfit), 1):
                yirData.iloc[t] = np.random.lognormal(bestfit.iat[t,0], variance, 1)

        curves = pd.concat([curves, yirData],axis=1)
        
        P = least_squares(ResidFun, inits, bounds=bounds, method=method,ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000000, args=(IC, timevect, conds, flag, yirData))
        Phats.iloc[simulation,:] = P["x"] # saves estimated parameters for each realization of a total of M realizations
    return Phats


##################### EXECUTIVE CODE #########################################

### inputs ######################
datafilename = "ebola1.csv"
data = pd.read_csv(datafilename, header=None)
DT=7

t_window = np.arange(0, 20)

cases = data.iloc[t_window, 1]

# conds
N=1000000
k=1/8
gam=1/6

conds=[N, k, gam];

# initial guesses
flag =3
beta0=2/3
q=1/2
v=0.5
inits=[beta0, q]

# initial conditions
I0 = cases.loc[0]
S0=N-I0
E0=0
R0=0
C0=I0
IC=[S0, E0, I0, E0, C0]

# bounds of params
bounds=([0, 0], [20, 1])


timevect= data.iloc[t_window,0] * DT

nsim=200

####### Run functions ##################################################
Ptrue = fitting_ggSEIR(cases, IC, inits, bounds, timevect, conds, flag, method='trf')
bestfit = pd.DataFrame(predicted(Ptrue, IC, timevect, conds, flag))
resid = bestfit.sub(cases, axis=0)

Phats = confint(Ptrue, bestfit, cases, IC, inits, bounds, timevect, conds, flag, nsim, dist=1, factor=None, method ='trf')

### Forecasting ##################
fct = 10
t_window2 = np.arange(0, t_window[-1]+fct)
timevect2 = t_window2 * DT

data2 = data.iloc[t_window2, :]
S_fc = pd.DataFrame([])
E_fc = pd.DataFrame([])
I_fc = pd.DataFrame([])
R_fc = pd.DataFrame([])
for simulation in range(nsim):
    params_hat = Phats.iloc[simulation, :]
    S, E, I, R, cum_incidence = solved_ggSEIR(params_hat, IC, timevect, conds, flag)
    S_fc = pd.concat([S_fc, S], axis=1)
    E_fc = pd.concat([E_fc, E], axis=1)
    I_fc = pd.concat([I_fc, I], axis=1)
    R_fc = pd.concat([R_fc, R], axis=1)

    