import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

class scenario:
    
    def __init__(self, A, N, SIR0, R0 = 2.2,T = 5.1, labels = None):
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
        if labels != None:
            self.num = dict(zip(labels, range(len(labels))))
        self.SIR = np.array([SIR0])
        
    def dSIR(self, SIR_snap):
        S,I,R = SIR_snap[0],SIR_snap[1],SIR_snap[2]
        
        def quant(X):
            return np.dot(self.A,X*self.Ninv) - self.Asum*X*self.Ninv
        
        dS = -self.beta*I*S*self.Ninv + quant(S)
        dI = self.beta*I*S*self.Ninv - self.gamma*I + quant(I)
        dR = self.gamma*I + quant(R)
        return np.array([dS,dI,dR])
        
    def march(self, nt):
        SIR = np.zeros((nt,3,self.A.shape[0]))
        SIR[0] = self.SIR[-1]
        
        for i in range(1,nt):
            SIR[i] = SIR[i-1]+self.dSIR(SIR[i-1])
        
        self.SIR = np.array(self.SIR.tolist() + SIR.tolist()[1:])
        
    def update_R(self, pairs):
        for (i,r) in pairs:
            self.beta[self.num[i]] = r*self.gamma[self.num[i]]
            
    def borders(self, pairs):
        for (c,b) in pairs:
            if b:
                self.A[self.num[c]] = self.A0[self.num[c]]
                self.Asum = np.sum(A,axis = 1)
            else:
                self.A[self.num[c]] = np.zeros_like(self.A[self.num[c]])
                self.Asum = np.sum(A,axis = 1)
        
    def plot(self, value = 1, as_percent = False):
        plt.figure(figsize=(16,10))
        for country in range(len(self.N)):
            s = np.array(self.SIR)[:,value,country]
            if as_percent:
                s = s*self.Ninv[country]
            plt.plot(range(len(s)),s)
        if self.labels != None:
            plt.legend(self.labels)
        plt.show()
                
    def for_vis(self, value = 1, as_json = True):
        vis = dict()
        
        for i in range(self.SIR.shape[0]):
            dt = dict(zip(self.labels,(self.SIR[i,1,:]*self.Ninv*100).astype(int).tolist()))
            vis[i] = dt
            
        return vis
            

def europe():
    Labels = ['BE','BG','CZ','DK','DE','EE','IE','EL','ES','FR','HR','IT',
      'CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH']
    N = [11590,6948,10709,5792,83784,1327,4938,10427,46755,65274,4105,60462,1170,1886,2722,
                         626,9660,442,17135,9006,37847,10197,19238,2078,5460,5541,10099,67886,5421,8655]
    num = dict(zip(Labels, range(len(Labels))))
    
    A = pd.read_csv("thematrix.csv" , header = None).values/(365*1000)
    SIR0 = np.array([N]+[[0]*len(N)]*2)
    inf = 100
    SIR0[:,num['IT']] = [N[num['IT']]-inf,inf,0]

    cs = scenario(A,N,SIR0,labels = Labels)
    cs.march(10)
    cs.borders([('DE',False)])
    cs.march(70)
    cs.plot(1, as_percent = True)
    
    return cs
    
def inter(day, day_old = 0, borders = None, SIR0 = None, max_days = 730):
    Labels = ['BE','BG','CZ','DK','DE','EE','IE','EL','ES','FR','HR','IT',
              'CY','LV','LT','LU','HU','MT','NL','AT','PL','PT','RO','SI','SK','FI','SE','UK','NO','CH']
    N = [11590,6948,10709,5792,83784,1327,4938,10427,46755,65274,4105,60462,1170,1886,2722,
         626,9660,442,17135,9006,37847,10197,19238,2078,5460,5541,10099,67886,5421,8655]
    num = dict(zip(Labels, range(len(Labels))))
    
    A = pd.read_csv("thematrix.csv" , header = None).values/(365*1000)

    if SIR0 == None:
        SIR0 = np.array([N]+[[0]*len(N)]*2)
        inf = 100
        SIR0[:,num['IT']] = [N[num['IT']]-inf,inf,0]

    cs = scenario(A,N,SIR0,labels = Labels)
    if day > day_old:
        cs.march(day-day_old)

    if borders != None:
        cs.borders([(x,False) for x in borders])

    cs.march(max_days - day)
    
    fur_martin = {"send_back": cs.SIR[day-day_old].tolist(), "frames": cs.for_vis()}
    
    return json.dumps(fur_martin)