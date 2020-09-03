import numpy as np
import pandas as pd
from scipy.stats import norm

def check_data(cluster,cid='cid',y='y'):
    clus = cluster.columns
    cl1 = len(clus)        
    if cl1!=2:
        if cl1<2:
            print("Not enough columns present. Less than 2 columns being present.")
            return False
        elif cl1>2:
            print("More columns present. More than 2 columns being present")
            return False
    else:
        if list(set(cluster[y].unique()))!=[0,1]:
            print("Binary class column is not consistent. Class labels are ", cluster[y].unique()) 
            return False
        if (cluster[y].dtype!=np.int64 and cluster[y].dtype!=np.float64 and cluster[y].dtype!=np.int32 and cluster[y].dtype!=np.float32):
            print("Binary class label is not Integer type")
            return False
    return True

def square(x):
    return x*x
def cube(x):
    return x*x*x
def fourthpower(x):
    return x*x*x*x

def normz(val,lower_tail=False):
    if lower_tail==False:
        return -norm.ppf(val)
    else:
        return norm.ppf(val)


def df_param(cluster,cid='ClusterId',y='Y'):
    k = len(cluster[cid].unique())
    ni = cluster.groupby(cid)[y].apply(lambda x: len(list(x)))
    N = sum(ni)
    yi = cluster.groupby(cid)[y].apply(lambda x: sum(x))
    return k,ni,N,yi

def kpr(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)
        pii = yi/ni
        wi = ni/N
        piw = sum(wi*pii)
        sw = sum(wi*square(pii - piw))
        rho_kpr = (sw - piw*(1 - piw)*sum(wi*(1 - wi)/ni))/(piw*(1 - piw)*(sum(wi*(1 - wi))) - sum(wi*(1 - wi)/ni))
        if ((rho_kpr < 0) | (rho_kpr > 1)):
            print("Warning: ICC Not Estimable by 'Moment with Weights Proportional to Cluster Size' Method")
        else:
            return rho_kpr   

def keq(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)    
        pii = yi/ni
        wi = np.repeat(1/k,k)
        piw = sum(wi*pii)
        sw = sum(wi * square(pii - piw))
        rho_keq = (sw - piw *(1 - piw)*sum(wi*(1 - wi)/ni))/(piw*(1 - piw)*(sum(wi*(1 - wi))) - sum(wi*(1 - wi)/ni))
        if ((rho_keq <0) | (rho_keq > 1)):
            print("Warning: ICC Not Estimable by 'Moment with Equal Weights' Method")
        else:
            return rho_keq 

#class moment_estimators:
def keqs(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)
        pii = yi/ni
        wi = np.repeat(1/k,k)
        piw = sum(wi * pii)
        sw = sum(wi * square(pii - piw))
        swn = (k - 1) * (sw/k)
        rho_keqs = (swn - piw*(1 - piw) * sum(wi*(1-wi)/ni))/(piw*(1 - piw)*(sum(wi*(1-wi))) - sum(wi*(1 - wi)/ni))
        if ((rho_keqs < 0) | (rho_keqs > 1)):
            print("Warning: ICC Not Estimable by 'Modified Moment with Equal Weights' Method")
        else:
            return rho_keqs

def kprs(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)
        pii = yi/ni
        wi = ni/N
        piw = sum(wi*pii)
        sw = sum(wi * square((pii - piw)))
        swn = (k-1)*sw/k
        rho_kprs = (swn - piw*(1-piw)*sum(wi*(1-wi)/ni))/(piw*(1-piw)*(sum(wi*(1-wi))) - sum(wi*(1-wi)/ni))  
        if ((rho_kprs < 0) | (rho_kprs > 1)):
            print("Warning: ICC Not Estimable by 'Modified Moment with Weights Proportional to Cluster Size Method")
        else:
            return rho_kprs        


def stab(cluster,cid='cid',y='y',kappa=0.45):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)
        n0 = (1/(k-1)) * (N - sum(square(ni)/N))
        p = sum(yi)/sum(ni)
        pii = yi/ni
        wi = ni/N
        piw = sum(wi*pii)
        sw = sum(wi*square(pii - piw))
        rho_stab = (1/(n0-1))*((N*sw)/((k-1)*p*(1-p)) + kappa - 1)
        if ((rho_stab < 0) | (rho_stab > 1)):
            print("Warning: ICC Not Estimable by 'Stabilized Moment Method'")
        else:
            return rho_stab

def ub(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)
        n0 = (1/(k-1))*(N - sum(square(ni)/N))
        yisq = square(yi)
        msw = (1/(N-k))*(sum(yi) - sum(yisq/ni))
        rho_ub = 1 - (N*n0*(k-1)*msw)/(sum(yi)*(n0*(k-1) - sum(yi)) + sum(yisq))
        if((rho_ub < 0) | (rho_ub > 1)):
            print("Warning: ICC Not Estimable by 'Unbiased Estimating Equation' Method")
        else:
            return rho_ub

def aov(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)    
        n0 = (1/(k-1)) * (N - sum(square(ni)/N))
        yisq = square(yi)
        msb = (1/(k-1)) * (sum(yisq/ni) - (1/N) * (square(sum(yi))))
        msw = (1/(N-k)) * (sum(yi) - sum(yisq/ni))
        rho_aov = (msb - msw) / (msb + (n0 -1) * msw)
        if ((rho_aov < 0) | (rho_aov > 1)):
            print(" Warning: ICC not estimable by ANOVA method")
        else:
            return rho_aov

def mod_aov(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y)        
        n0 = (1/(k-1)) * (N - sum(square(yi)/N))
        yisq = square(yi)
        msbs = (1/(k)) * (sum(yisq/ni) - (1/N)*(square(sum(yi))))
        msw = (1/(N-k)) * (sum(yi) - sum(yisq/ni))
        rho_mod_aov = (msbs - msw)/(msbs + (n0 - 1)*msw)
        if ((rho_mod_aov < 0) | (rho_mod_aov > 1)):
            print("Warning: ICC Not Estimable by 'Modified Anova Method'")
        else:
            return rho_mod_aov

def fleiss_cuzick(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y) 
        piio = sum(yi)/sum(ni)
        rho_fc = 1 - (1/((N-k) * piio * (1 - piio))) * sum(yi*(ni - yi)/ni)
        if ((rho_fc < 0) | (rho_fc > 1)):
            print("Warning: ICC Not Estimable by 'Fleiss-Cuzick's Kappa' Method")
        else:
            return rho_fc

def mak(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y) 
        yisq = square(yi)
        rho_mak = 1 - (k-1)*sum((yi*(ni - yi))/(ni*(ni -1))) / (sum(yisq/(square(ni))) + sum(yi/ni)*(k - 1 - sum(yi/ni)))
        if ((rho_mak < 0) | (rho_mak > 1)):
            print("Warning: ICC Not Estimable by 'Mak's Unweighted' Method")
        else:
            return rho_mak

def peq(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y) 
        mu_peq = sum((ni - 1)*yi)/sum((ni-1)*ni)
        rho_peq = (1/(mu_peq*(1 - mu_peq))) * (sum(yi* (yi - 1))/sum(ni*(ni - 1)) - square(mu_peq))
    #    rho_peq = (1/(mu_peq*(1 - mu_peq))) * (sum(yi*(yi - 1))/sum(ni(*ni - 1)) - square(mu_peq))
        if ((rho_peq < 0) | (rho_peq > 1)):
            print("Warning: ICC not Estimable by 'Correlation Method with Weight to Every pair of Observation'")
        else:
            return rho_peq


def pgp(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y) 
        mu_pgp = sum(yi/ni)/k    
        rho_pgp = (1/(mu_pgp*(1 - mu_pgp)))*(sum((yi*(yi-1))/(ni*(ni - 1)))/k - square(mu_pgp))
        if ((rho_pgp < 0) | (rho_pgp > 1)):
           print("Warning: ICC Not Estimable by 'Correlation Method with Equal Weight to each cluster irrespective of size'")
        else:
            return rho_pgp


def ppr(cluster,cid='cid',y='y'):
    if check_data(cluster,cid=cid,y=y):
        k,ni,N,yi = df_param(cluster,cid=cid,y=y) 
        mu_ppr = sum(yi)/N
        rho_ppr = (1/(mu_ppr * (1 - mu_ppr)))*(sum(yi*(yi - 1)/(ni - 1))/N - square(mu_ppr))
        if((rho_ppr < 0) | (rho_ppr > 1)):
            print("Warning: ICC Not Estimable by 'Correlation Method with weighting each pair according to Number of pairs individuals appear")
        else:
            return rho_ppr

def aov_conf_interval(rho_aov,zalpha,n0):
    st0 = 2 * square(1 - rho_aov)/square(n0)
    st1 = square(1 + rho_aov*(n0 - 1))/(N - k)
    st2 = ((k-1) * (1 - rho_aov) * (1 + rho_aov*(2*n0 - 1)) +
           square(rho_aov) * (sum(square(ni)) - (2/N) * sum(ni*ni*ni) + (1/square(N))*square(sum(square(ni)))))/square(k-1)
    var_smith_rho_aov = st0*(st1 + st2)
    ci_smith_rho_aov = [(rho_aov - zalpha * np.sqrt(var_smith_rho_aov)),(rho_aov + zalpha * np.sqrt(var_smith_rho_aov))]
    l_ci = ci_smith_rho_aov[0] < 0 and 0 or ci_smith_rho_aov[0]
    u_ci = ci_smith_rho_aov[1] > 1 and 0 or ci_smith_rho_aov[1]
    print(l_ci,u_ci)
    return l_ci,u_ci

def wald_conf_interval(rho_aov,yi,N,k):
    piio = sum(yi)/N
    lamb_da = (N - k)*(N - 1 - n0*(k - 1))*(rho_aov) + (N*(k - 1)*(n0 - 1))
    t0_zd = square((k - 1)*n0*N*(N - k))/fourthpower(lamb_da)
    t1_zd = 2*k + (1/(piio*(1 - piio)) - 6)*sum(1/ni)
    t2_zd = ((1/(piio*(1 - piio)) - 6)*sum(1/ni) - 2*N + 7*k - (8*square(k))/N - (2*k*(1 - k/N))/(piio*(1 - piio)) +
        (1/(piio*(1 - piio)) - 3)*sum(square(ni)))*rho_aov        
    t3_zd = (square(N) - square(k))/(piio*(1 - piio)) - 2*N - k + (4*square(k))/N + (7 - 8*k/N - (2*(1 - k/N))/(piio*(1 - piio)))*sum(square(ni))*square(rho_aov)
    t4_zd = (1/(piio*(1 - piio)) - 4)*(square((N - k)/N))*(sum(square(ni)) - N)*cube(rho_aov)                                                               
    var_zd_rho_aov = t0_zd *(t1_zd + t2_zd + t3_zd + t4_zd)                                                  
    ci_zd_rho_aov = [(rho_aov - zalpha*(np.sqrt(var_zd_rho_aov))),(rho_aov + zalpha*(np.sqrt(var_zd_rho_aov)))]
    l_ci = ci_zd_rho_aov[0] < 0 and 0 or ci_zd_rho_aov[0]
    u_ci = ci_zd_rho_aov[1] > 1 and 0 or ci_zd_rho_aov[1]
    print(l_ci,u_ci)
    return l_ci,u_ci

# +
# def fc_conf_interval(rho_fc,N,k,piio):
#     to_fc = 1 - rho_fc   
#     t1_fc = (1/(piio*(1 - piio)) - 6)*(sum(1/ni)/square(N - k)) + (2*N + 4*k - (k/(piio*(1 - piio))))*(k/(N*(square(N - k)))) + (sum(square(ni)))/(square(N)*piio*(1 - piio)) - ((3*N - 2*k)*(N-2*k)*sum(square(ni)))/(square(N)*(square((N-k))) - (2*N - k)/square((N - k)) * rho_fc)
#     t2_fc = ((4 - 1/(piio*(1 - piio))) * (sum(square(ni)))/square(N))))*square(rho_fc)
#     var_rho_fc = t0_fc*(t1_fc + t2_fc)
#     ci_rho_fc = [(rho_fc - zalpha*np.sqrt(var_rho_fc)), (rho_fc + zalpha*np.sqrt(var_rho_fc))]
#     l_ci = ci_rho_fc[0] < 0 and 0 or ci_rho_fc[0]
#     u_ci = ci_rho_fc[1] > 1 and 0 or ci_rho_fc[1]
#     return l_ci,u_ci
