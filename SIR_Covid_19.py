#COVID-19 SIR Model

from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import random


solutions = []

S0 = 10000
I0 = 1
R0 = 0
k = 1
gamma = .2

N=10000

tmax = 50
time_scale = np.linspace(0,tmax,tmax*24)
     
def SIR_model(f,t,k,gamma,N):
    S,I,R = f
    
    dS_dt = -k*S/N*I
    dI_dt = k*S/N*I - gamma*I
    dR_dt = gamma*I
    
    return([dS_dt, dI_dt, dR_dt])

for i in range(0,tmax):
    infection_chance = 10*random.random()

    if infection_chance > 2:
        R0 = R0 + 1
        I0 = I0 + 1
        S0 = S0
        solution = integrate.odeint(SIR_model,[S0,I0,R0], time_scale, args=(k,gamma,N))
        solutions = np.array(solution)
    else:
        I0 = I0 - 1
        S0 = S0 + 1
        R0 = R0
        solution = integrate.odeint(SIR_model,[S0,I0,R0], time_scale, args=(k,gamma,N))
        solutions = np.array(solution)        


plt.figure(figsize = [10,5])
plt.plot(time_scale,solution[:,0],label="S(t)")
plt.plot(time_scale,solution[:,1],label="I(t)")
plt.plot(time_scale,solution[:,2],label="R(t)")
plt.legend()
plt.xlabel("Time (Days)")
plt.ylabel("Number of people")
plt.title("SIR Model COVID-19")
plt.show()