import tensorflow as tf 
import numpy as np 
import neural_networks
import matplotlib.pyplot as plt
import sys, getopt
from utils import give_batch
import config as C
import time    
import para_fun as P
import base_function as BF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#problem = poisson_problem.poisson_2d()
    
class the_net():
    def __init__(self,train_config,stru_config):
        print("now initialize the net with para:")
        for item,value in train_config.items():
            print(item)
            print(value)
            print("======================")            
        self.save_path=train_config["CKPT"]
        self.learning_rate=train_config["LEARNING_RATE"]
        self.batch_size = train_config["BATCHSIZE"]#500
        self.max_iter = train_config["MAX_ITER"]
        self.epoch_save=train_config["EPOCH_SAVE"]
        self.step_each_iter=train_config['STEP_EACH_ITER']
        self.step_show=train_config['STEP_SHOW']
        self.global_steps = tf.Variable(0, trainable=False)  
        self.stru_config=stru_config     
###############################
#P.para
##############################
        self.sess=tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,#互相之间的多线程：parallelism：并行
            intra_op_parallelism_threads=1,#内部多线程
            ))
        print("openning sess")
        self.build_net()
        print("building net")
        
        
        self.build_opt()
        print("building opt")
        self.saver=tf.train.Saver(max_to_keep=1)#取最大值，max_to_keep：保持参数极值个数
        self.initialize()#初始化
        print("net initializing")
        #self.saver=tf.train.Saver(max_to_keep=1)
        self.D=give_batch([P.a,P.b])#限定区间[-1,1]
    def build_net(self):    
        para_dict=P.para_dict#获取对位字典
        equations_coe,value,coe_ret=BF.generate_poly(para_dict)#获取g（x）的系数矩阵
        self.stru_config["coe"]=coe_ret
        self.stru_config["right_power"]=para_dict["d"]-1#1
        self.stru_config["left_power"]=para_dict["d"]-1#1
        self.stru_config["xa"]=para_dict["left"]#a=-1
        self.stru_config["xb"]=para_dict["right"]#b=1
        self.y = neural_networks.neural_network(self.stru_config)#实例化stru_config
        
        
        
       # w = tf.get_variable("this_test", initializer=tf.constant([0.0001/(10**2)],tf.float64),dtype=tf.float64)
        #self.loss=self.y.input**2*w
        #self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_steps)#寻找梯度下降最大点以优化神经网络
       # exit()
        
        
        #print(self.y.value)
        #exit()
        
        self.sai0=tf.placeholder(tf.float64, [None, self.stru_config['n_inp']])
        self.sai1=tf.placeholder(tf.float64, [None, self.stru_config['n_inp']])#定义占位符(@形参)，行None列1
        #self.sai3=tf.placeholder(tf.float64, [None, self.stru_config['n_inp']])#定义占位符(@形参)，行None列1
        self.sai2=tf.placeholder(tf.float64, [None, self.stru_config['n_inp']])
        self.sai3=tf.placeholder(tf.float64, [None, self.stru_config['n_inp']])
        self.phi=tf.placeholder(tf.float64, [None, self.stru_config['n_inp']])
        #self.loss_name=self.y.d_values[8]+self.sai*self.y.value-self.phi#误差
        self.loss_name=self.y.d_values[4]+self.sai3*self.y.d_values[3]+self.sai2*self.y.d_values[2]+self.sai1*self.y.d_values[1]+self.sai0*self.y.value-self.phi#误差
        #print(self.y.d_values)
        #exit()
        #print(self.loss_name)
        #exit()
        

        self.loss_= tf.square(self.loss_name)#计算误差的平方
        print(self.loss_)
        #exit()
        
        self.loss=tf.sqrt(tf.reduce_mean(self.loss_)) #计算损失
        
        '''
    def build_opt(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_steps)
        '''

    def build_opt(self):
        #优化版梯度下降法寻找极值#minimize方法：最大限度地最小化 loss,tf.train.AdamOptimizer()函数是Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
        #decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)  
        #self.learning_rate_d = tf.train.exponential_decay(self.learning_rate,
        #                                   global_step=self.global_steps,
        #                                   decay_steps=self.step_each_iter,decay_rate=0.9)
        #print(self.loss)
        #exit()
        print("here1")
        print(self.loss)
        
       #self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_steps)#寻找梯度下降最大点以优化神经网络
        self.opt=tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,global_step=self.global_steps)#寻找梯度下降最大点以优化神经网络
        
        
        
        print("here")
        '''
    def opt(self):
        with tf.name_scope('optimizer'):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)
        '''
    def initialize(self):#初始化
        ckpt=tf.train.latest_checkpoint(self.save_path)#自动找到最近保存的变量文件
        if ckpt!=None:
            self.saver.restore(self.sess,ckpt)#无则存
            print("init from ckpt ")
        else:
            self.sess.run(tf.global_variables_initializer())#初始化
    def plot(self,value,real):#绘图
        plt.plot(value)
        plt.plot(real)
        plt.legend(["from MLP","exact solution"])
        plt.show()
    def train(self):
        st=time.time()#获得当前时间的时间戳（1970纪元后经过的浮点秒数）
        for epoch in range(self.max_iter):#遍历最大时间段，epoch时点
            print("train epoch %s of total %s epoches"%(epoch,self.max_iter))          
            for step in range(self.step_each_iter):#遍历 步
                intx=self.D.inter(self.batch_size)#随机数，500 x 1矩阵，内容其实就是输入数字
                #print(intx)
                sai0=P.give_sai0(intx)
                sai1=P.give_sai1(intx)
                sai2=P.give_sai2(intx)
                sai3=P.give_sai3(intx)
                phi=P.give_phi(intx)#f(x)标准函数的矩阵
                #\:转义到下一行继续
                loss,_,gs=self.sess.run([self.loss,self.opt,self.global_steps], \
                                            feed_dict={self.y.input:intx,self.sai0:sai0,self.sai1:sai1,self.sai2:sai2,self.sai3:sai3,self.phi:phi})#self.sai:sai,开始运行,self.sai1:sai1   sai3             
                if (step+1)%self.step_show==0:
                    #value=self.sess.run([self.phi.value], feed_dict={self.phi.input:intx, self.phi.bound:boundx})
                    print("loss %s,\
                    in epoch %s, in step %s \n, \
                    in global step %s, learning rate is %s, taks %s seconds"%(loss,epoch,step,gs,self.learning_rate,time.time()-st))
                    st=time.time()  
            if (epoch+1)%self.epoch_save==0:
                self.saver.save(self.sess, self.save_path+"/check.ckpt")
                int_x=[[x/100.0] for x in range(P.a*100,P.b*100)]
                real=P.give_exact(int_x)
                #sai=P.give_sai(int_x)
                sai0=P.give_sai0(intx)
                sai1=P.give_sai1(intx)#-1的矩阵
                sai2=P.give_sai2(intx)
                sai3=P.give_sai3(intx)
                phi=P.give_phi(int_x)
                value=self.sess.run(self.y.value, feed_dict={self.y.input:int_x,self.sai0:sai0,self.sai1:sai1,self.sai2:sai2,self.sai3:sai3,self.phi:phi})#,self.sai:sai,,,self.sai1:sai1,
                self.plot(value,real)
                print("Model saved in path: %s in epoch %s. learning_rate is %s" % (self.save_path,epoch,self.learning_rate))


if __name__ == '__main__':
    main_net=the_net(C.train_config,C.stru_config)
    main_net.train()
