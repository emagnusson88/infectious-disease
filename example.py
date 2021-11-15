import pandas as pd
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
from scipy.stats import poisson
from scipy.stats import nbinom
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from pandas.plotting import autocorrelation_plot

def SimpleGrowth(x, t, flag, params):
    '''
    Differential equation governing the growth
    flag: #1-Exp; 2-GGM; 3-Logistic Growth; 4-GLM
    '''
    if flag==1:
        r = params
        dx = r * x
    elif flag==2:
        r, p = params
        dx = r * (x**p)
    elif flag==3:
        r, K = params
        dx = r * x * (1 - (x/K))
    elif flag==4:
        r, p, K = params
        dx = r * (x**p) * (1 - (x/K))
    return dx


def solvedSimpleGrowth(params, IC, tf, flag):
    '''
    This function solves the differential equation in SimpleGrowth
    Parameters
    ----------
    params : a vector of parameters to be optimized
    IC : initial condition of the system
    tf : time vector
    flag : indicate what growth model should be solved

    Returns
    -------
    y : a vector of predicted case incidence

    '''
    x = pd.DataFrame(odeint(SimpleGrowth, IC, tf, args=(flag, params)))
    y = x.diff() #taking predicted incidence from 
    y.iat[0,0] = x.iat[0,0]
    return y


def ResidFun(params, IC, tf, flag, cases):
    '''
    Calculate residual = predicted - actual

    Parameters
    ----------
    params : a vector of parameters to be optimized
    IC : initial condition of the system
    tf : time vector
    flag : indicate what growth model should be solved
    cases : vector of actual case incidence

    Returns
    -------
    resid: a vector of residuals

    '''
    y = solvedSimpleGrowth(params, IC, tf, flag)
    z = y.sub(cases, axis=0)
    resid  = (z.values).ravel()
    return resid

def fittingSimpleGrowth(cases, inits, bounds, timevect, flag, method='trf'):
    '''
    Fitting model by Least Square curve fitting - minimize L2-norm
    
    Parameters
    ----------
    cases: vector of incidence
    timevect: vector of time points
    method: least_squares fitting method
    
    Returns
    -------
    Ptrue: a vector of estimated parameters after model fitting
    '''
    IC = cases.loc[0]
    P = least_squares(ResidFun, inits, bounds=bounds,method=method, ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000000, args=(IC, timevect, flag, cases))
    Ptrue = P["x"]
    return Ptrue


def confint(Ptrue, bestfit, cases, inits, bounds, timevect, flag, nsim, dist, factor=None, method ='trf'):
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
    IC = cases.loc[0]
    
    for simulation in range(nsim):
        yirData = pd.DataFrame(np.zeros((len(bestfit), 1)))
        yirData.iat[0,0] = bestfit.iat[0,0]
        
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
        P = least_squares(ResidFun, inits, bounds=bounds, args=(IC, timevect, flag, yirData))
        Phats.iloc[simulation,:] = P["x"] # saves estimated parameters for each realization of a total of M realizations
    return Phats

### inputs ######################
datafilename = "ebola1.csv"
data = pd.read_csv(datafilename, header=None)
#data = pd.read_csv(datafilename, sep="", header=None)

#data = data.drop(data.columns[[0,1,2,4,5]], axis=1) # to drop multiple columns created by blank_space in text file

flag=2 #1-Exp; 2-GGM; 3-Logistic Growth; 4-GLM

DT=7

t_window = np.arange(0, 20)
# an array from 0 to 19, equal to np.arange(20)

cases = data.iloc[t_window, 1]


timevect= data.iloc[t_window,0] * DT
vline_coord = timevect.tail(1)
 #timevect = data1[data1.columns[0:1]] * DT

# initial guess of parameters
r = 0.5
p = 0.5
K = 1000
if flag == 1:
    #r = raw_input("r: ")
    inits = r
elif flag == 2:
    #r = raw_input("r: ")
    #p = raw_input("p: ")
    inits = [r,p]
elif flag == 3:
    #r = raw_input("r: ")
    #K = raw_input("K: ")
    inits = [r, K]
elif flag == 4:
    #r = raw_input("r: ")
    #p = raw_input("p: ")
    #K = raw_input("K: ")
    inits = [r, p, K]

#initial condition
IC = cases.loc[0]

bounds=([0, 0], [100, 1])

####### Run functions ##################################################
Ptrue = fittingSimpleGrowth(cases, inits, bounds, timevect, flag, method='trf')
bestfit = pd.DataFrame(solvedSimpleGrowth(Ptrue, IC, timevect, flag))
resid = bestfit.sub(cases, axis=0)

nsim=200

#Poisson error structure
Phats = confint(Ptrue, bestfit, cases, inits, bounds, timevect, flag, nsim=nsim, dist=1, method='trf')
# Negative binomial error structure
#Phats = confint(Ptrue, bestfit, cases, inits, bounds, timevect, flag, nsim=nsim, dist=2,  factor=2,method='trf')
param_r = [np.mean(Phats.iloc[:,0]), np.percentile(Phats.iloc[:,0], 2.5), np.percentile(Phats.iloc[:,0], 97.5)]
param_p = [np.mean(Phats.iloc[:,1]), np.percentile(Phats.iloc[:,1], 2.5), np.percentile(Phats.iloc[:,1], 97.5)]

###### Visualization####################################################
plt.figure()
qqplot(resid) #normality

plt.figure()
autocorrelation_plot(resid) # autocorrelation

# Histogram of parameters in simulations
cad1 = "r=%s (95CI: %s,%s)" % ("%.2f" % param_r[0], "%.2f" % param_r[1], "%.2f" % param_r[2])
cad2 = "p=%s (95CI: %s,%s)" % ("%.2f" % param_p[0], "%.2f" % param_p[1], "%.2f" % param_p[2])
f, axarr = plt.subplots(2)
f.suptitle("Histogram of parameter", fontsize=16)
axarr[0].hist(Phats.iloc[:,0])
axarr[0].set_title(cad1)
axarr[0].set_ylabel("Frequency")

axarr[1].hist(Phats.iloc[:,1])
axarr[1].set_title(cad2)
axarr[1].set_ylabel("Frequency")

#axarr[1,0].hist(Phats.iloc[:,2])
#axarr[1,0].set_title(cad3)
#axarr[1,0].set_ylabel("Frequency")

### Forecasting ##################
fct = 10
t_window2 = np.arange(0, t_window[-1]+fct)
timevect2 = t_window2 * DT

data2 = data.iloc[t_window2, :]
curvesfc = pd.DataFrame([])

for simulation in range(nsim):
    params_hat = Phats.iloc[simulation, :]
    incidence = pd.DataFrame(solvedSimpleGrowth(params_hat, IC, timevect2, flag))
    curvesfc = pd.concat([curvesfc, incidence], axis=1)

vline_coord = timevect.tail(1)
fig_3 = plt.figure()
plt.plot(timevect2, curvesfc, 'c', linewidth=0.5)
plt.plot(timevect2, curvesfc.quantile(0.5, axis=1),'r-', linewidth=1)
plt.plot(timevect2, curvesfc.quantile(0.025, axis=1),'r--', linewidth=1)
plt.plot(timevect2, curvesfc.quantile(0.975, axis=1),'r--', linewidth=1)
plt.plot(timevect2,data2.iloc[:,1],'ko', linewidth=1)
[plt.axvline(x = i) for i in vline_coord]

plt.show(fig_3)