from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root,fsolve
#plt.rc('text', usetex=True) #使用latex
## 使用scipy.optimize模块的root和fsolve函数进行数值求解方程
 
## 1、求解f(x)=2*sin(x)-x+1
rangex1 = np.linspace(-2,8)
rangey1_1,rangey1_2 = 2*np.sin(rangex1),rangex1-1
plt.figure(1)
plt.plot(rangex1,rangey1_1,'r',rangex1,rangey1_2,'b--')
plt.title('$2sin(x)$ and $x-1$')
 
def f1(x):
 return np.sin(x)*2-x+1
 
sol1_root = root(f1,[2])
sol1_fsolve = fsolve(f1,[2])
plt.scatter(sol1_fsolve,2*np.sin(sol1_fsolve),linewidths=9)
plt.show()
 
## 2、求解线性方程组{3X1+2X2=3;X1-2X2=5}
def f2(x):
 return np.array([3*x[0]+2*x[1]-3,x[0]-2*x[1]-5])
 
sol2_root = root(f2,[0,0])
sol2_fsolve = fsolve(f2,[0,0])
print(sol2_fsolve) # [2. -1.5]
 
a = np.array([[3,2],[1,-2]])
b = np.array([3,5])
x = np.linalg.solve(a,b)
print(x) # [2. -1.5]
## 3、求解非线性方程组
def f3(x):
 return np.array([2*x[0]**2+3*x[1]-3*x[2]**3-7,
     x[0]+4*x[1]**2+8*x[2]-10,
     x[0]-2*x[1]**3-2*x[2]**2+1])
 
sol3_root = root(f3,[0,0,0])
sol3_fsolve = fsolve(f3,[0,0,0])
print(sol3_fsolve)
 
## 4、非线性方程
def f4(x):
 return np.array(np.sin(2*x-np.pi)*np.exp(-x/5)-np.sin(x))
init_guess =np.array([[0],[3],[6],[9]])
sol4_root = root(f4,init_guess)
sol4_fsolve = fsolve(f4,init_guess)
print(sol4_fsolve)
t = np.linspace(-2,12,2000)
y1 = np.sin(2*t-np.pi)*np.exp(-t/5)
y2 = np.sin(t)
plt.figure(2)
a , = plt.plot(t,y1,label='$sin(2x-\pi)e^{-x/5}$')
b , = plt.plot(t,y2,label='$sin(x)$')
plt.scatter(sol4_fsolve,np.sin(sol4_fsolve),linewidths=8)
plt.title('$sin(2x-\pi)e^{-x/5}$ and $sin(x)$')
plt.legend()
