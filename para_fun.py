import numpy as np
e=2.71828183
a1=-1.0/3.0
a2=4.0/9.0
def give_sai0(x):#零阶的系数
    ret=[]
    for xx in x:
        xx=xx[0]
        ret.append([-1.0])
    return ret
    
def give_sai1(x):#一阶的系数
    ret=[]
    for xx in x:
        xx=xx[0]
        ret.append([3.0*xx])

    return ret
    
def give_sai2(x):#二阶的系数
    ret=[]
    for xx in x:
        xx=xx[0]
        ret.append([e**xx])

    return ret
    
def give_sai3(x):#三阶的系数
    ret=[]
    for xx in x:
        xx=xx[0]
        ret.append([np.sin(xx)])

    return ret
    
def give_phi(x):#非齐次项
    ret=[]
    for xx in x:
        xx=xx[0]
        #ret.append([8.0*(2.0*xx*np.sin(xx)-7.0*np.cos(xx))])
        ret.append([-2.0*e**xx*(np.sin(xx)+np.cos(xx))+2.0*e**xx*(np.sin(xx)*np.cos(xx)-np.sin(xx)*np.sin(xx))+2.0*e**(2.0*xx)*np.cos(xx)+3.0*xx*e**xx*(np.sin(xx)+np.cos(xx))+2.0*xx-e**xx*np.sin(xx)])
    return ret

a=-1
b=1

A0=-1*e**(-1)*np.sin(1)-1
A1=e**(-1)*(np.cos(1)-np.sin(1))+1
A2=-4*np.sin(1)+2*np.cos(1)
A4=8*np.sin(1)-12*np.cos(1)
A6=-12*np.sin(1)+30*np.cos(1)
B0=e*np.sin(1)+1
B1=e*(np.sin(1)+np.cos(1))+1
B2=-4*np.sin(1)+2*np.cos(1)
B4=8*np.sin(1)-12*np.cos(1)
B6=-12*np.sin(1)+30*np.cos(1)




para_dict={"left":a,#对位字典
               "right":b,
               "left_order_value":[[0,A0],[1,A1]],
               "right_order_value":[[0,B0],[1,B1]],
               "d":4,#最大阶数-1=ma/mb
              }

'''
para_dict={"left":a,#对位字典
               "right":b,
               "left_order_value":[[0,A0],[2,A2],[4,A4],[6,A6]],
               "right_order_value":[[0,B0],[2,B2],[4,B4],[6,B6]],
              }
'''

def give_exact(x):#确切解
    ret=[]
    for xx in x:
        xx=xx[0]
        ret.append(e**xx*np.sin(xx)+xx)
    return ret

if __name__=="__main__":
    import matplotlib.pyplot as plt 
    x=[[xx/100.0]for xx in range(1000)]
    y=give_phi(x)
    plt.plot(y)
    plt.show()
