# Copyright 2020 Zhouchichun
# 59338158@qq.com
"""
    构建一个前馈神经网络 MLP
    给出值，一阶，二阶，三阶 导数，暂时没有混合导数 
    方法如下：phi=neural_network(config)
    phi.value      phi.value_bound
    phi.d_value    phi.d_value_bound
    phi.dd_value   phi.dd_value_bound
    phi.input      phi.input_bound
"""
import tensorflow as tf 
import time
import Legend#图例
import config as C
import para_fun as P
import base_function as BF

def get_activate(act_name):
    if act_name=="sigmoid":
        return tf.nn.sigmoid
    elif act_name=="tanh":
        return tf.nn.tanh
    elif act_name=="relu":
        return tf.nn.relu
    else:
        print("激活函数配置错误，请检查config.py文件，激活函数从['tanh','relu','sigmoid']")
        exit()
        
class neural_network: 

    def __init__(self,config):

        self.n_input = config['n_inp']
        self.struc =  config['struc']
        self.var_name = config['var_name']
        self.coe=config['coe']
        self.right_power=config['right_power']
        self.left_power=config['left_power']
        self.xa=config['xa']
        self.xb=config['xb']
        self.order_up=config['order_up']
        self.order_down=config['order_down']
        self.highest   =config["derive_order"]
        
        self.n_output = 1
        self.weight_initialization =  tf.contrib.layers.xavier_initializer()#一种带权重的初始化方法
        
        ######
        print('建立MLP网络，参数如下')
        for ke,va in config.items():
            print(ke,va)
            print('---------------------------------------------')
        ######

        self.construct_input()
        self.build_value_leg()
        self.build_derivation()
        


    def get_activate(self,act_name):
        if act_name=="sigmoid":
            return tf.nn.sigmoid
        elif act_name=="tanh":
            return tf.nn.tanh
        elif act_name=="relu":
            return tf.nn.relu
        else:
            print("激活函数配置错误，请检查config.py文件，激活函数从['tanh','relu','sigmoid']")
            exit()

#############搭建MLP指定层数，指定每一曾的神经元个数
    def construct_input(self):
        print("建立placeholder")
        self.input=tf.placeholder(tf.float64, [None, self.n_input])
    def build_bound(self):#限定Netx
    
        #g=1
        for ll,c in enumerate(self.coe):
            if ll==0:
                g=tf.constant(c)
                continue
            g +=c*self.input**ll
    
        self.value=self.value*(self.input-self.xa)**self.left_power*(self.input-self.xb)**self.right_power\
        +g 
    def build_value(self):
        print("建立网络结构")
        self.value_up=0
        self.value_down=0
        for i in range(1,self.order_up+1):
            w = tf.get_variable(self.var_name + 'weight_up' + str(i), 
                                initializer=tf.constant([0.01/i],tf.float64),
                                dtype=tf.float64)
            tmp=w*tf.pow(self.input,i)
            self.value_up +=tmp
        for i in range(1,self.order_down+1):
            w = tf.get_variable(self.var_name + 'weight_down' + str(i), 
                                initializer=tf.constant([0.01/i],tf.float64),
                                dtype=tf.float64)
            tmp=w*tf.pow(self.input,i)
            self.value_down +=tmp
        self.value=tf.divide(self.value_up,self.value_down)
        self.build_bound()
        
    def build_value_mlp(self):
        
        print("建立网络结构")
        for i,stru in enumerate(self.struc):
            this_num,this_act=stru
            activate=get_activate(this_act)
            if i == 0:
                w = tf.get_variable(self.var_name + 'weight_' + str(0), 
                                    shape=[self.n_input, this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.input, w), b))
               
            else:
                w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                    shape=[self.struc[i-1][0], this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                b = tf.get_variable(self.var_name + 'bias_' + str(i), 
                                    shape=[this_num], 
                                    initializer=self.weight_initialization, 
                                    dtype=tf.float64)
                self.layer = activate(tf.add(tf.matmul(self.layer, w), b))
                
        w =  tf.get_variable(self.var_name+'weight_' + str(len(self.struc)), 
                            shape=[self.struc[-1][0], self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        b =tf.get_variable(self.var_name+'bias_' + str(len(self.struc)), 
                            shape=[self.n_output], 
                            initializer=self.weight_initialization, 
                            dtype=tf.float64)
        self.value=tf.matmul(self.layer, w) + b
        self.build_bound()
    def build_value_leg(self):
        print("建立网络结构")
        self.value=0
        self.order=6
        self.legends=Legend.give_legend(self.input,self.order)#勒让德多项式拟合
        for i in range(1,self.order+1):
            w = tf.get_variable(self.var_name + 'weight_' + str(i), 
                                initializer=tf.constant([0.0001/(i**2)],tf.float64),
                                dtype=tf.float64)
            self.value +=w*self.legends[i]
        self.build_bound()
#############
    
    def build_derivation(self):
        print("建立导数")
       # self.highest=8
        
        self.d_values={}
        for i in range(1,self.highest+1):
            st=time.time()
            print("建立导数 %s"%i)
            if i==1:
                 self.d_values[i]=tf.gradients(self.value,self.input)[0]
                 print("SSSSSSSSSSSSS")
                 print(self.value)
            else:
                 self.d_values[i]=tf.gradients(self.d_values[i-1],self.input)[0]
                 print("DDDDDDDDDDDDD")
                 print(self.d_values[i])
            print("用时 %s"%(time.time()-st))
            st=time.time()
         
if __name__=="__main__":
    
    stru_config=C.stru_config
    para_dict=P.para_dict
    #coe_ret=[ 6.76562681e-01 1.64672578e-18 -5.68068741e-01 -1.04796463e-18 -1.16982020e-01 -1.68250891e-19 8.48807966e-03  4.38939609e-20]
    _,_,coe_ret=BF.generate_poly(para_dict)
    stru_config["coe"]=coe_ret
    stru_config["right_power"]=para_dict["d"]-1
    stru_config["left_power"]=para_dict["d"]-1
    stru_config["xa"]=para_dict["left"]
    stru_config["xb"]=para_dict["right"]
    sess=tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=1,#互相之间的多线程：parallelism：并行
        intra_op_parallelism_threads=1,#内部多线程
        ))
    y=neural_network(stru_config)

    value=sess.run(y)
    #y.build_value_leg()

