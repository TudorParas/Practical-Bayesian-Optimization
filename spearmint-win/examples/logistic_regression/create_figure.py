from numpy import genfromtxt
import numpy as np
import matplotlib.pyplot as plt

def getmin(data):
    res=[]
    n=np.shape(data)[0]
    for i in range(n):
        if(data[i,-1]==100):
            res.append(data[i,1])
            break
        if(data[i,-1]==0):
            continue
        if(data[i,-1]!=data[i+1,-1]):
            res.append(data[i,1])
    return(res)


my_data_mcmc1 = genfromtxt('logitreg1\\trace.csv', delimiter=',')
my_data_mcmc2 = genfromtxt('logitreg2\\trace.csv', delimiter=',')
my_data_mcmc3 = genfromtxt('logitreg3\\trace.csv', delimiter=',')
my_data_mcmc4 = genfromtxt('logitreg4\\trace.csv', delimiter=',')
my_data_mcmc5 = genfromtxt('logitreg5\\trace.csv', delimiter=',')
my_data_mcmc6 = genfromtxt('logitreg6\\trace.csv', delimiter=',')
my_data_mcmc7 = genfromtxt('logitreg7\\trace.csv', delimiter=',')
my_data_mcmc8 = genfromtxt('logitreg8\\trace.csv', delimiter=',')
my_data_mcmc9 = genfromtxt('logitreg9\\trace.csv', delimiter=',')
my_data_mcmc10 = genfromtxt('logitreg10\\trace.csv', delimiter=',')

res_mcmc1=np.asarray(getmin(my_data_mcmc1))
res_mcmc2=np.asarray(getmin(my_data_mcmc2))
res_mcmc3=np.asarray(getmin(my_data_mcmc3))
res_mcmc4=np.asarray(getmin(my_data_mcmc4))
res_mcmc5=np.asarray(getmin(my_data_mcmc5))
res_mcmc6=np.asarray(getmin(my_data_mcmc6))
res_mcmc7=np.asarray(getmin(my_data_mcmc7))
res_mcmc8=np.asarray(getmin(my_data_mcmc8))
res_mcmc9=np.asarray(getmin(my_data_mcmc9))
res_mcmc10=np.asarray(getmin(my_data_mcmc10))

final_mcmc = np.stack((res_mcmc1, res_mcmc2,res_mcmc3,res_mcmc4,
                       res_mcmc5,res_mcmc6,res_mcmc7,res_mcmc8,
                       res_mcmc9,res_mcmc10), axis=-1)
mu_mcmc=np.mean(final_mcmc,-1)
std_mcmc=np.std(final_mcmc,-1)


my_data_opt1 = genfromtxt('logitregOpt1\\trace.csv', delimiter=',')
my_data_opt2 = genfromtxt('logitregOpt2\\trace.csv', delimiter=',')
my_data_opt3 = genfromtxt('logitregOpt3\\trace.csv', delimiter=',')
my_data_opt4 = genfromtxt('logitregOpt4\\trace.csv', delimiter=',')
my_data_opt5 = genfromtxt('logitregOpt5\\trace.csv', delimiter=',')
my_data_opt6 = genfromtxt('logitregOpt6\\trace.csv', delimiter=',')
my_data_opt7 = genfromtxt('logitregOpt7\\trace.csv', delimiter=',')
my_data_opt8 = genfromtxt('logitregOpt8\\trace.csv', delimiter=',')
my_data_opt9 = genfromtxt('logitregOpt9\\trace.csv', delimiter=',')
my_data_opt10 = genfromtxt('logitregOpt10\\trace.csv', delimiter=',')

res_opt1=np.asarray(getmin(my_data_opt1))
res_opt2=np.asarray(getmin(my_data_opt2))
res_opt3=np.asarray(getmin(my_data_opt3))
res_opt4=np.asarray(getmin(my_data_opt4))
res_opt5=np.asarray(getmin(my_data_opt5))
res_opt6=np.asarray(getmin(my_data_opt6))
res_opt7=np.asarray(getmin(my_data_opt7))
res_opt8=np.asarray(getmin(my_data_opt8))
res_opt9=np.asarray(getmin(my_data_opt9))
res_opt10=np.asarray(getmin(my_data_opt10))

final_opt = np.stack((res_opt1, res_opt2,res_opt3,res_opt4,
                       res_opt5,res_opt6,res_opt7,res_opt8,
                       res_opt9,res_opt10), axis=-1)
mu_opt=np.mean(final_opt,-1)
std_opt=np.std(final_opt,-1)


my_data_sec1 = genfromtxt('logitregPersec1\\trace.csv', delimiter=',')
my_data_sec2 = genfromtxt('logitregPersec2\\trace.csv', delimiter=',')
my_data_sec3 = genfromtxt('logitregPersec3\\trace.csv', delimiter=',')
my_data_sec4 = genfromtxt('logitregPersec4\\trace.csv', delimiter=',')
my_data_sec5 = genfromtxt('logitregPersec5\\trace.csv', delimiter=',')
my_data_sec6 = genfromtxt('logitregPersec6\\trace.csv', delimiter=',')
my_data_sec7 = genfromtxt('logitregPersec7\\trace.csv', delimiter=',')
my_data_sec8 = genfromtxt('logitregPersec8\\trace.csv', delimiter=',')
my_data_sec9 = genfromtxt('logitregPersec9\\trace.csv', delimiter=',')
my_data_sec10 = genfromtxt('logitregPersec10\\trace.csv', delimiter=',')

res_sec1=np.asarray(getmin(my_data_sec1))
res_sec2=np.asarray(getmin(my_data_sec2))
res_sec3=np.asarray(getmin(my_data_sec3))
res_sec4=np.asarray(getmin(my_data_sec4))
res_sec5=np.asarray(getmin(my_data_sec5))
res_sec6=np.asarray(getmin(my_data_sec6))
res_sec7=np.asarray(getmin(my_data_sec7))
res_sec8=np.asarray(getmin(my_data_sec8))
res_sec9=np.asarray(getmin(my_data_sec9))
res_sec10=np.asarray(getmin(my_data_sec10))

final_sec = np.stack((res_sec1, res_sec2,res_sec3,res_sec4,
                       res_sec5,res_sec6,res_sec7,res_sec8,
                       res_sec9,res_sec10), axis=-1)
mu_sec=np.mean(final_sec,-1)
std_sec=np.std(final_sec,-1)


#Result of Tree Parzen Algorithm was obtained from running TPA.py
mu_tpe=np.array([0.72704, 0.65815, 0.65798, 0.5755 , 0.55624, 0.55624, 0.55624,
       0.55624, 0.53353, 0.51482, 0.46441, 0.45568, 0.44287, 0.44287,
       0.43435, 0.41907, 0.41907, 0.41907, 0.41907, 0.39285, 0.23494,
       0.22317, 0.22317, 0.22317, 0.19209, 0.1844 , 0.16952, 0.15701,
       0.14175, 0.14175, 0.14175, 0.14175, 0.14097, 0.14097, 0.13458,
       0.13458, 0.13458, 0.13458, 0.12106, 0.12106, 0.12106, 0.11841,
       0.11841, 0.11841, 0.11841, 0.11841, 0.11841, 0.11841, 0.11841,
       0.11841, 0.11841, 0.11671, 0.11671, 0.11671, 0.11671, 0.11354,
       0.11354, 0.11354, 0.11354, 0.11168, 0.11168, 0.11168, 0.11031,
       0.11031, 0.11031, 0.0968 , 0.09632, 0.09632, 0.09632, 0.09632,
       0.09632, 0.09632, 0.09515, 0.09374, 0.09374, 0.09374, 0.09374,
       0.09374, 0.09374, 0.09374, 0.08904, 0.08904, 0.08904, 0.08904,
       0.08904, 0.08904, 0.08904, 0.08904, 0.08904, 0.08904, 0.08904,
       0.08904, 0.08904, 0.08904, 0.08904, 0.08904, 0.08861, 0.08861,
       0.08861, 0.088  ])
    
std_tpe=np.array([0.23158388, 0.21556005, 0.21536881, 0.2165928 , 0.21936457,
       0.21936457, 0.21936457, 0.21936457, 0.19432473, 0.19022203,
       0.12306743, 0.11153901, 0.09675778, 0.09675778, 0.09741683,
       0.086059  , 0.086059  , 0.086059  , 0.086059  , 0.12327465,
       0.10470853, 0.10192416, 0.10192416, 0.10192416, 0.0825328 ,
       0.07937062, 0.04932222, 0.04905504, 0.0468855 , 0.0468855 ,
       0.0468855 , 0.0468855 , 0.04750461, 0.04750461, 0.04550186,
       0.04550186, 0.04550186, 0.04550186, 0.04108236, 0.04108236,
       0.04108236, 0.04188629, 0.04188629, 0.04188629, 0.04188629,
       0.04188629, 0.04188629, 0.04188629, 0.04188629, 0.04188629,
       0.04188629, 0.04187654, 0.04187654, 0.04187654, 0.04187654,
       0.03450354, 0.03450354, 0.03450354, 0.03450354, 0.03547686,
       0.03547686, 0.03547686, 0.03601895, 0.03601895, 0.03601895,
       0.01637803, 0.01655758, 0.01655758, 0.01655758, 0.01655758,
       0.01655758, 0.01655758, 0.01600314, 0.01665888, 0.01665888,
       0.01665888, 0.01665888, 0.01665888, 0.01665888, 0.01665888,
       0.0077643 , 0.0077643 , 0.0077643 , 0.0077643 , 0.0077643 ,
       0.0077643 , 0.0077643 , 0.0077643 , 0.0077643 , 0.0077643 ,
       0.0077643 , 0.0077643 , 0.0077643 , 0.0077643 , 0.0077643 ,
       0.0077643 , 0.00792735, 0.00792735, 0.00792735, 0.0067155 ])

plt.errorbar(list(range(1,101)), mu_mcmc, std_mcmc, marker='.',label='GP EI MCMC',color='black')
plt.errorbar(list(range(1,101)), mu_opt, std_opt, marker='.',label='GP EI Opt',color='blue')
plt.errorbar(list(range(1,101)), mu_sec, std_sec, marker='.',label='GP EI per Sec',color='orange')
plt.errorbar(list(range(1,101)), mu_tpe, std_tpe, marker='.',label='Tree Parzen Algorithm',color='green')
plt.xlabel('Function Evaluations')
plt.ylabel('Min Function Value')
plt.ylim(0.075,0.2)
plt.legend()


"""
The second figure is from the first figure, but the x-axis 
is the time elapsed.
"""