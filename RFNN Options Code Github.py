# -*- coding: utf-8 -*-
"""
@author: Lukas Gonon
"""


import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as normal
import os
from sklearn import linear_model
import scipy.linalg

class dataGenerator(object):
    """Generate (noisy) option price data from a Black-Scholes model"""
    def __init__(self, ex_dict=None):
        ## Model and parameter specifications, these can also be changed by 
        ## optional dictionary input (ex_dict) (see below) 
        
        self.d = 50
        self.n_time = 10000        
        self.T = 1.
        self.rho = 0.2
        self.sigma = 0.2 
        self.s0 = 1.
        self.K0 = 1.
        self.r = 0.
        self.n_samples = 200000
        self.n = 200000
        self.n_split = 20000

        ## MJD parameters
        self.sigmaMJD=0.5
        self.rhoMJD=0.2
        self.intensity=5.


        ## Time grid and horizon
        self.dtSim = (self.T+0.0)/self.n_time
        self.tSim = np.arange(0, self.n_time+1)*self.dtSim
        self.sqrt_dtSim = np.sqrt(self.dtSim)
        self.sqrt_T = np.sqrt(self.T)

        self.save_dir = os.getcwd() 
        
        ### Here any of the above are replaced by values in ex_dict
        if ex_dict != None:
            for param_name,param_value in ex_dict.items():
                setattr(self,param_name,param_value)

        self.strikes = normal.rvs(mean=self.K0,cov=20, size=self.n)
        self.initialvalues = np.random.uniform(-1.,1.,size=(1,self.d,self.n))
        self.Corr = np.ones([self.d,self.d])*self.rho+np.diag(np.diag(np.ones([self.d,self.d])))*(1.-self.rho)
        self.CorrMJD = np.ones([self.d,self.d])*self.rhoMJD+np.diag(np.diag(np.ones([self.d,self.d])))*(1.-self.rhoMJD)

        self.L = scipy.linalg.sqrtm(self.Corr)
        self.LMJD = scipy.linalg.sqrtm(self.CorrMJD)
     

    def check_sigma_cond(self):
        Sigma = self.Corr*(self.sigma)**2
        val,vec = np.linalg.eig(0.5*Sigma)
        return np.min(val)-1./(self.T*np.sqrt(8.)*np.pi) > 0.

    def BS_sample_exact(self,n_sample):
        B_T = np.reshape(normal.rvs(mean=np.zeros(self.d),cov=1, size=n_sample)*self.sqrt_T,(n_sample, self.d))
        W_T = np.dot(B_T,self.L.T)[:,:,np.newaxis]
        BS_sample_T = np.ones([n_sample, self.d, 1])*self.s0*np.exp((self.r-0.5*(self.sigma**2))*self.T)
        BS_sample_T = BS_sample_T * np.exp(self.sigma*W_T)
        return np.real(BS_sample_T)
    
    def MJD_sample_exact(self,n_sample):
        n_jumps = np.reshape(np.random.poisson(lam=self.intensity*self.T, size=n_sample),(n_sample, 1))
        sqrt_n_jumps = np.sqrt(n_jumps)
        B_T = np.reshape(normal.rvs(mean=np.zeros(self.d),cov=1, size=n_sample)*self.sqrt_T,(n_sample, self.d))
        W_T = np.dot(B_T,self.L.T)[:,:,np.newaxis]
        Z_T = np.reshape(normal.rvs(mean=np.zeros(self.d),cov=1, size=n_sample)*sqrt_n_jumps,(n_sample, self.d))
        C_T = np.dot(Z_T,self.LMJD.T)[:,:,np.newaxis]
        
        MJD_sample_T = np.ones([n_sample, self.d, 1])*self.s0*np.exp((self.r-0.5*(self.sigma**2)-self.intensity*(np.exp(0.5*(self.sigmaMJD**2))-1))*self.T)
        MJD_sample_T = MJD_sample_T * np.exp(self.sigma*W_T+self.sigmaMJD*C_T)
        return np.real(MJD_sample_T)        
    
    def prices_maxcall(self,label='',save_prices=True):
        samples = self.BS_sample_exact(self.n_samples)
        assetmax = np.amax(samples,1)
        strikes_reshaped = np.reshape(self.strikes,(1,self.n))
        #prices = np.mean(np.maximum(assetmax-strikes_reshaped,0),axis=0)
        split = np.split(strikes_reshaped,1000,axis=1)
        prices = np.zeros([self.n//1000,1000])
        strikes = np.zeros([self.n//1000,1000])
        for i in range(1000):
            if i<10 or (i/100).is_integer():
                print(i)
            prices[:,i] = self.compute_prices_maxcall(assetmax,split[i])
            strikes[:,i] = split[i]
        strikes = np.reshape(strikes.flatten(),[self.n,1])
        prices = np.reshape(prices.flatten(),[self.n,1])
        if save_prices:
            np.savez(self.save_dir+label+'.npz',strikes=strikes,prices=prices)      
        return strikes,prices
    
    def solPDE(self,label='',save_prices=True):
            samples = self.BS_sample_exact(self.n_samples)#/self.s0
            #samples = self.MJD_sample_exact(self.n_samples)
            split = np.split(self.initialvalues,self.n_split,axis=2)
            PDEsol = np.zeros([self.n//self.n_split,self.n_split])
            #xvalues = np.zeros([self.n//1000,1000])
            for i in range(self.n_split):
                if i<10 or (i/100).is_integer():
                      print(i)   
                PDEsol[:,i] = self.compute_PDEsol(samples,split[i])
                #xvalues[:,:,i] = split[i]
            #xvalues = np.reshape(xvalues.flatten(),[self.n,1])
            PDEsol = np.reshape(PDEsol.T,[self.n,1])
            if save_prices:
                np.savez(self.save_dir+label+'.npz',xvalues=self.initialvalues,PDEsol=PDEsol)      
            return self.initialvalues,PDEsol

    def compute_PDEsol(self,samples,initialvalues):
        #return np.mean(np.maximum(np.amax(samples*np.exp(initialvalues),axis=1)-self.K0,0),axis=0)
        #return np.mean(np.maximum(self.K0-np.amin(samples*np.exp(initialvalues),axis=1),0),axis=0)
        return np.mean(np.maximum(np.mean(samples*np.exp(initialvalues),axis=1)-self.K0,0),axis=0)
    
    def compute_prices_maxcall(self,maxima,strikes):
        return np.mean(np.maximum(maxima-strikes,0),axis=0)
        
    def load_PDEsol(self,label=''):
        npzfile = np.load(self.save_dir+label+'.npz')
        return npzfile['xvalues'], npzfile['PDEsol']

    def load_prices(self,label=''):
        npzfile = np.load(self.save_dir+label+'.npz')
        return npzfile['strikes'], npzfile['prices']
    
    def dW_sample(self, n_sample):
        dW_sample = np.zeros([n_sample, self.d, self.n_time])
        for i in range(self.n_time):
            dW_sample[:, :, i] = np.reshape(normal.rvs(mean=np.zeros(self.d),cov=1, size=n_sample)*self.sqrt_dtSim,(n_sample, self.d))
        return dW_sample
    

class RandomNeuralNetworkLearner(object):
    """Learn using random neural networks"""
    def __init__(self, ex_dict=None):
        ## Model and parameter specifications, these can also be changed by 
        ## optional dictionary input (ex_dict) (see below) 
        
        self.d = 1
        self.N = 50

        
        self.sigmoid = lambda x: np.maximum(x,0)
        self.nu = 5
        ### Here any of the above are replaced by values in ex_dict
        if ex_dict != None:
            for param_name,param_value in ex_dict.items():
                setattr(self,param_name,param_value)
                
        ## Random weights      
        self.A = np.random.normal(0.0,1.0,size=(self.d,self.N))*np.sqrt(self.nu)*np.sqrt(1./np.random.chisquare(self.nu, size=(1,self.N)))
        self.B = np.random.standard_t(2,size=(1,self.N))


    
    
    def train_random_features(self,X_samples,Y_samples,ridge=None):
        randomfeatures = self.sigmoid(np.dot(X_samples,self.A)+self.B)  
           
        linear = linear_model.LinearRegression(fit_intercept=False)#
        if ridge is not None:
            linear = linear_model.Ridge(fit_intercept=False,alpha=ridge)#
        self.model = linear.fit(randomfeatures,Y_samples)
        #print(linear.coef_)
        #print(linear.intercept_)
        
    
    def predict(self,X_test):
        randomfeatures = self.sigmoid(np.dot(X_test,self.A)+self.B)  
        return self.model.predict(randomfeatures)
    
    def prediction_error_random_features(self,X_test,Y_test):
        Y_predict = self.predict(X_test)
        return np.sqrt(np.mean((Y_predict-Y_test)**2))
    
    
    def plot(self,X_test,Y_test):
        plt.figure()
        f,p= plt.subplots(self.d,1,figsize=(6,6),sharey=True)
        for i in range(self.d):
            p[i].plot(self.model.predict(X_test)[:,i],'b',label=label+'test')
            p[i].plot(Y_test[:,i],'g',label='Test Path')
        plt.savefig(label + '.pdf')
        plt.legend()
        plt.show()
       
        


if __name__ == '__main__':
    d=50
    addlabel = '' #'MJDCase3' #'max_call'#min_put'
    datagenerator = dataGenerator({'d':d})
    n = datagenerator.n
    load_data=True
    if False:
        load_data = True
        if load_data:
            X_samples,Y_samples = datagenerator.load_prices('train')
            X_test, Y_test = datagenerator.load_prices(label='test')    
        else:        
            X_samples,Y_samples = datagenerator.prices_maxcall('train')
            datagenerator = dataGenerator({'n':10000})
            X_test, Y_test = datagenerator.prices_maxcall(label='test')    
        learner = RandomNeuralNetworkLearner({'N':10})
        learner.train_random_features(X_samples, Y_samples)
        print(learner.prediction_error_random_features(X_test,Y_test))
    
    
    if load_data:
        X_samples,Y_samples = datagenerator.load_PDEsol('trainPDE'+str(d)+addlabel)
        X_test, Y_test = datagenerator.load_PDEsol(label='testPDE'+str(d)+addlabel)
    else:
        X_samples,Y_samples = datagenerator.solPDE('trainPDE'+str(d)+addlabel)
        datagenerator = dataGenerator({'n':50000,'n_split':1000,'d':d})
        X_test, Y_test = datagenerator.solPDE(label='testPDE'+str(d)+addlabel)     


    N_N = 20
    err = np.zeros([N_N,1])
    Nlist = np.array([x*10 for x in range(N_N)])
    Nlist[0] =1
    nlist = N_N*[n]
    for i in range(N_N):
        learner = RandomNeuralNetworkLearner({'N':Nlist[i],'d':d})
        learner.train_random_features(X_samples.T[0:nlist[i],:,0], Y_samples[0:nlist[i]])#/np.sqrt(Nlist[i]))
        err[i] = learner.prediction_error_random_features(X_test.T[:,:,0],Y_test)
    
    fig, ax = plt.subplots()
    plt.plot(Nlist,err,'o:',label='estimated error')
    plt.plot(Nlist,err[0,0]*np.sqrt(Nlist[0]/Nlist),label='theoretical decay')
    plt.legend(['estimated error','theoretical decay'])
    plt.xlabel(r'$N$')
    plt.ylabel('Learning Error')  
    plt.savefig('plot'+str(d)+addlabel+'.pdf')    
    plt.show()

    start_ind = 0
    linear = linear_model.LinearRegression(fit_intercept=True)#
    linear.fit(np.reshape(np.log(Nlist[start_ind:]),[len(Nlist[start_ind:]),1]),np.log(err[start_ind:]))
    print(linear.coef_)
    print(linear.intercept_)
    
    
    fig, ax = plt.subplots()
    #ax.set_title('Learining Error')
    plt.plot(np.log(Nlist),np.log(err),'o:',label='log(estimated error)')
    plt.plot(np.log(Nlist),np.log(err[0,0]*np.sqrt(Nlist[0]/Nlist)),label='theoretical decay (log-scale))')
    plt.plot(np.log(Nlist),(np.log(Nlist)*linear.coef_).T+linear.intercept_,label='regression line')
    plt.legend(['log(estimated error)','theoretical decay (log-scale)','regression line'])
    plt.xlabel(r'$log(N)$')
    plt.ylabel('Learning Error (log-scale)')  
    plt.savefig('loglog'+str(d)+addlabel+'.pdf')    
    plt.show()
    

   
    
    NRuns = 50
    err2 = np.zeros([N_N,NRuns])
    #Nlist = np.array([1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190])
    #Nlist = np.array([100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290])  
    #Nlist = np.array([500,510,520,530,540,550,560,570,580,590])
    for i in range(N_N):
        for j in range(NRuns):
            learner = RandomNeuralNetworkLearner({'N':Nlist[i],'d':d})
            learner.train_random_features(X_samples.T[0:nlist[i],:,0], Y_samples[0:nlist[i]])#,ridge=0.1)
            err2[i,j] = learner.prediction_error_random_features(X_test.T[:,:,0],Y_test)
    errAvg = np.mean(err2,axis=1)
    fig, ax = plt.subplots()
    #ax.set_title('Learining Error')
    plt.plot(Nlist,errAvg,label='error')
    plt.plot(Nlist,errAvg[0]*np.sqrt(Nlist[0]/Nlist),label='sqrt')
    plt.legend(['estimated error','theoretical decay'])
    plt.xlabel(r'$N$')
    plt.ylabel('Learning Error')  
    plt.savefig('plotAvg'+str(d)+addlabel+'.pdf')    
    plt.show()
        
    start_ind = 0
    linear = linear_model.LinearRegression(fit_intercept=True)#
    linear.fit(np.reshape(np.log(Nlist[start_ind:]),[len(Nlist[start_ind:]),1]),np.log(errAvg[start_ind:]))
    print(linear.coef_)
    print(linear.intercept_)
    
    plot_ind=0
    fig, ax = plt.subplots()
    #ax.set_title('Learining Error')
    plt.plot(np.log(Nlist[plot_ind:]),np.log(errAvg[plot_ind:]),'o:',label='log(estimated error)')
    plt.plot(np.log(Nlist[plot_ind:]),np.log(errAvg[plot_ind]*np.sqrt(Nlist[plot_ind]/Nlist[plot_ind:])),label='theoretical decay (log-scale))')
    plt.plot(np.log(Nlist[plot_ind:]),(np.log(Nlist[plot_ind:])*linear.coef_).T+linear.intercept_,label='regression line')
    plt.legend(['log(estimated error)','theoretical decay (log-scale)','regression line'])
    plt.xlabel(r'$log(N)$')
    plt.ylabel('Learning Error (log-scale)')  
    plt.savefig('loglogplotAvg'+str(d)+addlabel+'.pdf')    
    plt.show()
    