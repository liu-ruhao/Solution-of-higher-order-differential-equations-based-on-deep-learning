import numpy as np 
import matplotlib.pyplot as plt
import sys
import random



class give_batch():
    def __init__(self, x_range):
        self.x_range = x_range

    def inter(self, batchsize):
        ret = np.random.uniform(self.x_range[0], self.x_range[1], batchsize)#在-1到1之间随机采样
        ret=np.reshape(ret,[batchsize,1])#修改ret为 500 x 1的矩阵
        return ret

    def bound(self, batchsize):
        ret=np.array([self.x_range[0]]*batchsize)     
        ret=np.reshape(ret,[batchsize,1])   
        return ret
    def bound_r(self, batchsize):
        ret=np.array([self.x_range[1]]*batchsize)     
        ret=np.reshape(ret,[batchsize,1])   
        return ret


if __name__ == '__main__':
    D=give_batch([0,10])
    plt.plot(D.inter(100))
    plt.show()
    plt.plot(D.bound(100))
    plt.show()
    plt.plot(D.bound_r(100))
    plt.show()
