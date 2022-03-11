import numpy as np
import matplotlib.pyplot as plt
import para_fun as P
def factor(n):#阶乘
    ret=1
    for i in range(1,n+1):
        ret *=i
    return ret

def generate_poly(para_dict):#获得8x1矩阵（列表）
    '''
    para_dict:{"left":-2,
               "right":2,
               "left_order_value":[[0,A0],[2,A2],[4,A4]],
               "right_order_value":[[0,B0],[2,B2],[4,B4]],
              }
    '''
    '''
    para_dict={"left":a,
                   "right":b,
                   "left_order_value":[[0,A0],[2,A2],[4,A4],[6,A6]],
                   "right_order_value":[[0,B0],[2,B2],[4,B4],[6,B6]],
                  }
    '''
    max_order=4#len(para_dict["left_order_value"])+len(para_dict["right_order_value"])#2
    coe={}
    left=para_dict["left"]
    right=para_dict["right"]
    for i in range(max_order):#i~(0-7)#获得基本列表
        tmp_left=[]#每个i循环清空一次列表
        tmp_right=[]#同上
        for j in range(max_order):#j~(0-7)
            if j<i:
                tmp_left.append(0)#+0
                tmp_right.append(0)
            elif j==i:
                tmp_left.append(factor(j))#+j！
                tmp_right.append(factor(j))
            else:#j>i
                p=j-i#j-i
                c=factor(j)/factor(j-i)#Ⅱ(j-k){k~[0,i-1]},A j取i,记为c
                tmp_left.append(c*left**p)#c((-1)^p)
                tmp_right.append(c*right**p)#c[(1)^p]
        coe["left_%s"%i]=tmp_left#i~(0-7),j=7
        coe["right_%s"%i]=tmp_right
    equations_coe=[]
    value=[]
    for order,v in  para_dict["left_order_value"]:
        equations_coe.append(coe["left_%s"%order])#获得i=0,2,4,6时的左列表
        value.append(v)#获得列表值
    for order,v in  para_dict["right_order_value"]:
        equations_coe.append(coe["right_%s"%order])
        value.append(v)
    coe_matrix=np.array(equations_coe)#获得8x8数组矩阵coe_matrix
    coe_ret=np.dot(np.linalg.inv(coe_matrix),value)#获得8x1矩阵（列表），np.dot：计算矩阵或向量内积;np.linalg.inv()：矩阵求逆
    print(equations_coe)
    print(value)
    print(coe_ret)
    return equations_coe,value,coe_ret

def plot_poly(coes,a,b):
    x=np.linspace(a,b,100)#获得[a,b]间均匀分布样本100个
    y=0
    for i,c in enumerate(coes):
        y +=c*x**i#y=y+cx^i
    ex=give_exact(x)#[(x^2-1)cosx]
    plt.plot(-y)#-y,ex都是列表，横坐标默认为间隔数
    plt.plot(ex)
    plt.legend(["poly","exact"])
    plt.show()
def check(coes,a,b,order):
    pass
    
def give_exact(x):#[(x^2-1)cosx]
    ret=[]
    for xx in x:
        #xx=xx[0]
        ret.append((xx**2.0-1.0)*np.cos(xx))#(x^2-1)cosx
    return ret
if __name__=="__main__":
    a=-1
    b=1
    #sin1=0.8414709848078965,cos1=0.5403023058681398,sin1-cos1=0.30116867893975674
    A0=0
    A2=-4*np.sin(1)+2*np.cos(1)#-4sin1+2cos1=-2.2852793274953065
    A4=8*np.sin(1)-12*np.cos(1)#8sin1-12cos1=0.24814020804549486
    A6=-12*np.sin(1)+30*np.cos(1)#-12sin1+30cos1=6.111417358349435
    B0=0
    B2=-4*np.sin(1)+2*np.cos(1)
    B4=8*np.sin(1)-12*np.cos(1)
    B6=-12*np.sin(1)+30*np.cos(1)
    para_dict=P.para_dict
    equations_coe,value,coe_ret=generate_poly(para_dict)
    c_r=[6.76562681e-01,1.64672578e-18,-5.68068741e-01,-1.04796463e-18,-1.16982020e-01,-1.68250891e-19,8.48807966e-03,4.38939609e-20]
    print(type(c_r))
    plot_poly(coe_ret,para_dict["left"],para_dict["right"])