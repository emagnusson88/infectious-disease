def SEIQRDV(y, t, params, conds, flag):

#time-dependent beta(t): exponential decline

#fixed conditions
    N=conds[0]
    k= conds[1]
    gamma=conds[2]

#parameters
    beta0=params[0]
   #k=params(2)
   # gam=params[2]


#differential equations
    S, E, I, Q, R, D, V, C = y
    dSdt = -beta * S * I / N - alpha*S
    dEdt = beta * S * I / N +sigma*beta*V - k*E
    dIdt = k*E - gamma * I
    dQdt = gamma*I - (1-p)*lamb*Q - p*rho*Q
    dRdt = (1-p)*lamb*Q 
    dDdt = p*rho*Q
    dVdt = alpha*S - sigma*beta*V
    dCdt = k*E #cumulative incidence of infectious

    return dSdt, dEdt, dIdt, dRdt, dCdt


# Integrate the SIR equations over the time grid, t.
def solved_SEIQRDV(params, y, t, conds, flag):
    ret = odeint(SEIQRDV, y, t, args=(params, conds, flag))
    S, E, I, Q, R, D, V, cum_incidence= ret.T
    incidence = pd.Series(cum_incidence).diff()
    incidence.iat[0] = cum_incidence[0]
    return incidence

def ResidFun(params, IC, tf,  conds, flag, cases):
    y = solved_SEIQRDV(params, IC, tf, conds, flag)
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
    
