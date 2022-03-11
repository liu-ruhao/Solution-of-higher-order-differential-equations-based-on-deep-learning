import numpy as np

def give_y(x):
    ret=[]
    for xx in x:
        xx=xx[0]
        ret.append(1.0/4*(np.exp(xx)-np.exp(-xx))-1.0/2*np.sin(xx))

    return ret
if __name__=="__main__":
    import matplotlib.pyplot as plt 
    x=[[xx/100.0]for xx in range(1000)]
    y=give_y(x)
    plt.plot(y)
    plt.show()
