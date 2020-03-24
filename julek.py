# Author: Julian Wykowski jw6717@ic.ac.uk

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class scenario:

    # Initialise a simulation
    def __init__(self, A, N, SIR0, labels, R0 = 2.2,T = 5.1):
        # A is adjacency matrix, N array with population of each state
        # SIR0 the initial state of S,I and R, labels a list of state names
        self.A = np.array(A).astype("double")
        self.A0 = A
        self.Asum = np.sum(A,axis = 1)
        self.N = np.array(N).astype("double")
        self.Ninv = np.reciprocal(self.N)
        self.R = R0
        self.T = T
        self.beta = np.array([R0/T]*len(N))
        self.gamma = np.array([1/T]*len(N))
        self.labels = labels
        self.num = dict(zip(labels, range(len(labels))))
        self.SIR = np.array([SIR0])

    # Calculate the derivative of SIR at point t given SIR(t)
    def dSIR(self, SIR_snap):
        S,I,R = SIR_snap[0],SIR_snap[1],SIR_snap[2]
        
        def quant(X):
            W = X*self.Ninv
            return self.A.dot(W) - self.Asum*W
        
        dS = -self.beta*I*S*self.Ninv + quant(S)
        dI = self.beta*I*S*self.Ninv - self.gamma*I + quant(I)
        dR = self.gamma*I + quant(R)
        return np.array([dS,dI,dR])

    # Perform time-march for nt days 
    def march(self, nt):
        if nt > 0:
            SIR = np.zeros((nt+1,3,self.A.shape[0]))
            SIR[0] = self.SIR[-1]
            
            for i in range(1,nt+1):
                SIR[i] = SIR[i-1]+self.dSIR(SIR[i-1])
            
            self.SIR = np.append(self.SIR,SIR[1:], axis = 0)

    # Update the value of R for pairs = [(label, R_value), ...]    
    def update_R(self, pairs):
        for (i,r) in pairs:
            self.beta[self.num[i]] = r*self.gamma[self.num[i]]

    # Update the border situation   
    def closed_borders(self, countries = []):
        # Sets all borders open except those that are passed in array countries
        self.A = self.A0.copy()

        for c in countries:
            self.A[self.num[c],:] = np.zeros_like(self.A[self.num[c],:])
            self.A[:,self.num[c]] = np.zeros_like(self.A[:,self.num[c]])
        
        self.Asum = np.sum(self.A,axis = 1)
    
    # Run time march for 730 days, updating borders and/or R at appropriate events
    def full_run(self, events = {}, max_days = 730, R_range = (1.1,2.2)):
        # Expects events to be a dict, where events[t] = {"R": pairs for update_R() at t,
        # "closed_borders": countries for closed_borders() at t}
        time = [int(i) for i in events.keys()] + [max_days-1]
        self.march(time[0])

        for i in range(len(time)-1):
            event = events[str(time[i])]

            if "closed_borders" in event.keys():
                self.closed_borders(event["closed_borders"])
            if "R" in event.keys():
                rd = events[str(time[i])]["R"]
                self.update_R([(i,(int(rd[i])*((R_range[1]-R_range[0])/100))+1.1) for i in rd.keys()])

            self.march(time[i+1]-time[i])
        
    # Create graph for current simulation
    def plot(self, value = 1, as_percent = False):
        plt.figure(figsize = (10,7))
        for country in range(len(self.N)):
            s = np.array(self.SIR)[:,value,country]
            if as_percent:
                s = s*self.Ninv[country]
            plt.plot(range(len(s)),s)
        if self.labels != None:
            plt.legend(self.labels)
        plt.show()

    # For website visualisation       
    def for_vis(self, value = 1, as_json = True):
        mp = dict()
        
        for i in range(self.SIR.shape[0]):
            mp[i] = dict()
            for j in range(self.SIR.shape[2]):
                mp[i][self.labels[j]] = {"infected_percentage": \
                    int(self.SIR[i,1,j]*100*self.Ninv[j]), "infected_total": \
                        int(self.SIR[i,1,j]*1000), "recovered_total": int(self.SIR[i,2,j]*1000)}
        
        pl = (self.SIR[:,1,:]*self.Ninv).tolist()
        fur_martin = {"plot": pl, "map": mp}
        if as_json: fur_martin = json.dumps(fur_martin)
        return fur_martin
            
# Load europe scenario
def europe(SIR0 = None):
    Labels = ['BE','BG','CZ','DK','DE','EE','IE','EL','ES','FR','HR','IT',
      'CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH']
    N = [11590,6948,10709,5792,83784,1327,4938,10427,46755,65274,4105,60462,1170,1886,2722,
                         626,9660,442,17135,9006,37847,10197,19238,2078,5460,5541,10099,67886,5421,8655]
    num = dict(zip(Labels, range(len(Labels))))
    A = pd.read_csv("backend/thematrix.csv" , header = None).values/(1000)

    df = pd.read_csv("backend/SIR0.csv")
    df = df.loc[df["Month"] == 3].loc[df["Day"] == 12]
    SIR0 = np.array([N]+[[0]*len(N)]*2)
    for i in Labels:
        SIR0[1,num[i]] = df.loc[df["Label"] == i,"Cases"].iloc()[0]/1000
        SIR0[0,num[i]] -= SIR0[1,num[i]]

    cs = scenario(A,N,SIR0,Labels)
    return cs

# Interface with website
def inter(events = {}, SIR0 = None, as_json = True, max_days = 730):
    cs = europe(SIR0)
    cs.full_run(events, max_days = max_days)
    return cs.for_vis(as_json)

# Demonstration
def demo():
    cs = europe()
    cs.closed_borders(["DE"])
    cs.march(30)
    cs.closed_borders()
    cs.march(100)
    cs.plot(as_percent = True)