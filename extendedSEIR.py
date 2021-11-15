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

def extendedSEIR(y, t, params, conds, flag):
    '''
    beta : transmission rate
    alpha : vaccination rate
    1/k : incubation period
    sigma : vaccination efficacy
    1/gamma : infectious period
    p : case fatality rate
    1/lamb : recovery time
    rho : time until death
    pa : proportion of undiagnosed cases
    
    S : susceptible
    E : exposed
    I : infectious (diagnosed)
    A : infectious (undiagnosed) - have to make assumption of proportion of COVID cases undiagnosed
    Q : quarantine
    R : recovered
    D : Dead
    V : Vaccinated

    Parameters
    ----------
    y : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    conds : TYPE
        DESCRIPTION.
    flag : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    #time-dependent beta(t): exponential decline
    if flag==1:
        #fixed conditions
        N, alpha, k, sigma, p, lamb, rho, gamma = conds
        #parameters
        beta = params        
        #differential equations
        S, E, I, Q, R, D, V, C = y
        dSdt = -beta * S * I / N - alpha*S
        dEdt = beta * S * I / N +sigma*beta*V - k*E
        dVdt = alpha*S - sigma*beta*V
        dIdt = k*E - gamma * I
        dQdt = gamma*I - (1-p)*lamb*Q - p*rho*Q
        dRdt = (1-p)*lamb*Q 
        dDdt = p*rho*Q       
        dCdt = k*E #cumulative incidence of infectious
        return dSdt, dEdt, dIdt, dQdt, dRdt, dDdt, dVdt, dCdt
    
    elif flag==2:
        #fixed conditions
        N, alpha, k, sigma, p, lamb, rho, gamma1, gamma2, pa = conds
        
        S, E, I, A, Q, R, D, V, C = y
        dSdt = -beta * S * (I+A) / N - alpha*S
        dEdt = beta * S *(I+A)/ N +sigma*beta*V - k*E
        dVdt = alpha*S - sigma*beta*V
        dIdt = k*(1-pa)*E - gamma1 * I
        dAdt = k*pa*E - gamma2*A
        dQdt = gamma1*I + gamma2*A- (1-p)*lamb*Q - p*rho*Q
        dRdt = (1-p)*lamb*Q 
        dDdt = p*rho*Q
        dCdt = k*(1-pa)*E #cumulative incidence of infectious
        return dSdt, dEdt, dIdt, dAdt, dQdt, dRdt, dDdt, dVdt, dCdt
    

# Integrate the SIR equations over the time grid, t.
def solved_extendedSEIR(params, y, t, conds, flag):
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
    ret = odeint(extendedSEIR, y, t, args=(params, conds, flag))
    if flag==1:
        S, E, I, Q, R, D, V, cum_incidence = ret.T   
        return S, E, I, Q, R, D, V, cum_incidence
    elif flag==2:
        S, E, I, A, Q, R, D, V, cum_incidence = ret.T   
        return S, E, I, A, Q, R, D, V, cum_incidence

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
    if flag == 1:
        S, E, I, Q, R, D, V, cum_incidence= solved_extendedSEIR(params, IC, tf, conds, flag)
    elif flag == 2:
        S, E, I, A, Q, R, D, V, cum_incidence= solved_extendedSEIR(params, IC, tf, conds, flag)
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


def fitting_extendedSEIR(cases, IC, inits, bounds, timevect, conds, flag, method='trf'):
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


    
