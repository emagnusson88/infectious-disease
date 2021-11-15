def ggSEIR(y, t, params, conds, flag):

#time-dependent beta(t): exponential decline

#fixed conditions
    N=conds[0]
    k= conds[1]
    gam=conds[2]

#parameters
    beta0=params[0]
   #k=params(2)
   # gam=params[2]


    if flag==1: #exponential decline
        q=params[2]
        phi=gam/beta0
        beta=beta0*((1-phi)*exp(-q*t) + phi)
    elif flag==2: #hyperbolic decline
        q=params[2]
        v=conds[4]
        phi=gam/beta0
        beta=beta0*((1-phi)*(1/(1+q*v*t)**(1/v)) + phi)
    elif flag==3: #harmonic decline
        q=params(2)
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
    ret = odeint(ggSEIR, y, t, args=(params, conds, flag))
    S, E, I, R, cum_incidence = ret.T
    incidence = pd.Series(cum_incidence).diff()
    incidence.iat[0] = cum_incidence[0]
    return incidence

def ResidFun(params, IC, tf,  conds, flag, cases):
    y = solved_ggSEIR(params, IC, tf, conds, flag)
    z = y.sub(cases, axis=0)
    return (z.values).ravel()


def fitting_ggSEIR(cases, inits, bounds, timevect, conds, flag, method):
    '''
    cases: vector of incidence
    timevect: vector of time points
    method: least_squares fitting method
    '''
    P = least_squares(ResidFun, inits, bounds=bounds, ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=1000000, args=(IC, timevect, conds, flag, cases))
    Ptrue = P["x"]
    return Ptrue
    






