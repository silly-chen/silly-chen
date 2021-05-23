import tensorflow as tf
import tflib as lib
import functools
import tensorflow.contrib.slim as slim
import ops
from tflib import layernorm
from tflib import batchnorm
from tflib import conv2d
import numpy as np
def LeakyReLU(x, alpha=0.2):#lrelu激活函数
    return tf.maximum(alpha*x, x)

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def ReLULayer(name, n_in, n_out, inputs):#全连接层经relu输出
    output = lib.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):#全连接层经lrelu输出
    output = lib.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Normalize(name, axes, inputs):#归一化
    if ('Discriminator' in name):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.batchnorm.Batchnorm(name,axes,inputs,fused=True)


# def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
#     output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
#     output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
#     return output

# def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
#     output = inputs
#     output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
#     output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
#     return output
def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs,he_init=True, biases=True):#卷积+平均池化
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.nn.avg_pool(output,ksize= [1, 1, 2, 2],strides= [1, 1, 2, 2],padding= 'SAME',data_format='NCHW',name=None)#步长为2，尺寸缩小一倍
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):#平均池化+卷积
    # print('inputs.shap:',inputs.shape)
    output = tf.nn.avg_pool(inputs,ksize= [1, 1, 2, 2],strides= [1, 1, 2, 2],padding= 'SAME',data_format='NCHW',name=None)
    # print('output.shap:',output.shape)
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output
#步长为2的卷积+卷积，用于下采样
def Conv2_conv(name, input_dim, output_dim, filter_size, inputs, k_h=3,k_w=3,he_init=True, biases=True,stddev=0.02):
    with tf.variable_scope(name):
        #步长为2的卷积
        w41=tf.get_variable('w41',[k_h,k_w,inputs.get_shape()[1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        output1=tf.nn.conv2d(inputs,w41,strides=[1,1,2,2],padding='SAME',data_format='NCHW')
        output=tf.nn.relu(output1)
        w52=tf.get_variable('w51',[k_h,k_w,output.get_shape()[1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        output2=tf.nn.conv2d(output,w52,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
    return output2
 #卷积+步长为2的卷积，用于下采样
def Conv_conv2(name, input_dim, output_dim, filter_size, inputs, k_h=3,k_w=3,he_init=True, biases=True,stddev=0.02):
    with tf.variable_scope(name):
        #步长为2的卷积
        output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
        w2=tf.get_variable('w2',[k_h,k_w,output.get_shape()[1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        output=tf.nn.conv2d(output,w2,strides=[1,1,2,2],padding='SAME',data_format='NCHW')
        output=tf.nn.relu(output)
         
    return output
 #5*5卷积      
def conv2d(input_, output_dim,
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv
#1*1卷积
def conv1x1(input_, output_dim,
              init=tf.contrib.layers.xavier_initializer(), name='conv1x1'):
  # print('conv1x1 input  shape',input_.shape)
  with tf.variable_scope(name):
    k_h = 1
    k_w = 1
    d_h = 1
    d_w = 1
    w3 = tf.get_variable(
        'w3', [k_h, k_w, input_.get_shape()[1], output_dim],
        initializer=init)
    # print('input_.get_shape()[1]:',input_.get_shape()[1],w)
    conv = tf.nn.conv2d(input_, w3, strides=[1, 1,d_h, d_w ], padding='SAME',data_format='NCHW')
    return conv
#反卷积
def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,  name="deconv2d"):
    with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
         w4 = tf.get_variable('w4', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    try:
            deconv = tf.nn.conv2d_transpose(input_, w4, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
            deconv = tf.nn.deconv2d(input_, w4, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv
#上采样+卷积
def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

#步长为2的卷积代替池化层的残差块，Down模式
def Res_conv_d2(name,input_dim,output_dim,filter_size,inputs,resample=None,he_init=True):
    if resample=='down':
        conv_shortcut = Conv2_conv
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(Conv_conv2, input_dim=input_dim, output_dim=output_dim) 
    if output_dim==input_dim :
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs) 
    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)
   
    return output+shortcut
#残差块
def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    if resample=='down':#下采样，池化+卷积
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':#上采样，反卷积
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:#不进行操作，卷积
        conv_shortcut = lib.conv2d.Conv2D
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # 第二分支
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)
    output = inputs#第一条支路
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)#1层conv+bn+relu
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)#2层con+bn+relu
    return shortcut + output

#模块1：上采样+变尺度空洞空间上下文+conv
def Upsample_Atrous_Conv_1(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True,stddev=0.02):
    with tf.variable_scope(name):
        output = inputs
        output = tf.concat([output, output, output, output], axis=1)
        output = tf.transpose(output, [0,2,3,1])
        output = tf.depth_to_space(output, 2)
        output1 = tf.transpose(output, [0,3,1,2])
        #3*3
        output=slim.conv2d(output1,output_dim,[3,3],stride=1,padding='SAME',data_format='NCHW',scope='con_3')
        #1*1
        output2=tf.transpose(output,(0,2,3,1))
        
        w18=tf.get_variable('w18',[1,1,output2.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1x1=tf.nn.atrous_conv2d(output2, w18, rate=[1,1],padding='SAME',name='conv1x1d')
        
        w19=tf.get_variable('w19',[3,3,output2.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3x3_1=tf.nn.atrous_conv2d(output2,w19,rate=[1,1],padding='SAME',name='conv3x3_1e')
        conv3x3_3=tf.nn.atrous_conv2d(output2,w19,rate=[2,2],padding='SAME',name='conv3x3_3r')
        conv3x3_5=tf.nn.atrous_conv2d(output2,w19,rate=[5,5],padding='SAME',name='conv3x3_4t')
        output=tf.concat([conv1x1,conv3x3_1,conv3x3_3,conv3x3_5],axis=-1)
        output=tf.transpose(output,(0,3,1,2))
        output=slim.conv2d(output,output_dim,[1,1],stride=1,padding='SAME',data_format='NCHW',scope='1x1_22')
        print('0out6',output.shape)

    return tf.nn.relu(output)
#模块2：上采样+变尺度空洞空间上下文+conv
def Upsample_Atrous_Conv_2(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True,stddev=0.02):
    with tf.variable_scope(name):
        output = inputs
        output = tf.concat([output, output, output, output], axis=1)
        output = tf.transpose(output, [0,2,3,1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0,3,1,2])
        #3*3
        output=slim.conv2d(output,output_dim,[3,3],stride=1,padding='SAME',data_format='NCHW',scope='con_33')
        #1*1
        output=tf.transpose(output,(0,2,3,1))
        w20=tf.get_variable('w20',[1,1,output.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv1x1=tf.nn.atrous_conv2d(output, w20, rate=[1,1],padding='SAME',name='conv1x1o')
        
        w21=tf.get_variable('w21',[3,3,output.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3x3_1=tf.nn.atrous_conv2d(output,w21,rate=[1,1],padding='SAME',name='conv3x3_1p')
        conv3x3_3=tf.nn.atrous_conv2d(output,w21,rate=[2,2],padding='SAME',name='conv3x3_3q')
        conv3x3_5=tf.nn.atrous_conv2d(output,w21,rate=[5,5],padding='SAME',name='conv3x3_5r')
        output=tf.concat([conv1x1,conv3x3_1,conv3x3_3,conv3x3_5],axis=-1)
        print('out2',output.shape)#48,64,256
        output=tf.transpose(output,(0,3,1,2))
        output=slim.conv2d(output,output_dim,[1,1],stride=1,padding='SAME',data_format='NCHW',scope='1x1w')
        print('out3',output.shape)#64,48,64
    return tf.nn.relu(output)

#残差连接而成的变尺度空洞空间上下文模块，跳跃连接使用
def Atrous_Conv_res(name,inputs, input_dim, output_dim, filter_size,k_h=3,k_w=3,he_init=True,biases=True,stddev=0.02):
    with tf.variable_scope(name):
        output=inputs
        output=Normalize(name+'BN',[0,2,3],output)
        output=tf.nn.relu(output)
        print('out0',output.shape)
        #3*3卷积
        output1=slim.conv2d(output,output_dim,[3,3],stride=1,padding='SAME',data_format='NCHW',scope='conv9')
        #1*1卷积
        output2=tf.transpose(output1,(0,2,3,1))
        w5=tf.get_variable('w5',[1,1,output2.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))        
        conv1x1=tf.nn.atrous_conv2d(output2, w5, rate=[1,1], padding='SAME', name='conv1x1')
        #3*3空洞卷积，是否需要转置？
        w6=tf.get_variable('w6',[3,3,output2.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3x3_1=tf.nn.atrous_conv2d(output2, w6, rate=[1,1], padding='SAME', name='conv3x3_1')              
        conv3x3_2=tf.nn.atrous_conv2d(output2, w6, rate=[2,2], padding='SAME', name='conv3x3_2')
        conv3x3_3=tf.nn.atrous_conv2d(output2, w6, rate=[5,5], padding='SAME', name='conv3x3_3')      
        output=tf.concat([conv1x1,conv3x3_1,conv3x3_2,conv3x3_3],axis=-1)
        
        #1*1卷积
        output=tf.transpose(output,(0,3,1,2))
        output=slim.conv2d(output,output_dim,[1,1],stride=1,padding='SAME',data_format='NCHW',scope='conv1x1_2')
        
        shortcut=slim.conv2d(inputs,output_dim,[3,3],stride=1,padding='SAME',data_format='NCHW',scope='shortcut-1')
        return output+shortcut    
    
#conv+变尺度空洞空间上下文模块
def Conv_Atrous(name,input_dim,output_dim,filter_size,inputs,k_h=3,k_w=3,he_init=True,biases=True,stddev=0.02):
    with tf.variable_scope(name):
        #3*3卷积
        output=slim.conv2d(inputs,output_dim,[3,3],stride=1,padding='SAME',data_format='NCHW',scope='conv')
        #1*1卷积
        output=tf.transpose(output,(0,2,3,1))
        w7=tf.get_variable('w7',[1,1,output.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))        
        conv1x1=tf.nn.atrous_conv2d(output, w7, rate=[1,1], padding='SAME', name='conv1x1-')
        #3*3空洞卷积，是否需要转置？
        w8=tf.get_variable('w8',[3,3,output.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3x3_1=tf.nn.atrous_conv2d(output, w8, rate=[1,1], padding='SAME',name='conv3x3_1-')              
        conv3x3_2=tf.nn.atrous_conv2d(output, w8, rate=[2,2], padding='SAME', name='conv3x3_2-')
        conv3x3_3=tf.nn.atrous_conv2d(output, w8, rate=[5,5], padding='SAME', name='conv3x3_3-')      
        output=tf.concat([conv1x1,conv3x3_1,conv3x3_2,conv3x3_3],axis=-1)
        
        #1*1卷积
        output=tf.transpose(output,(0,3,1,2))
        output=slim.conv2d(output,output_dim,[1,1],stride=1,padding='SAME',data_format='NCHW',scope='conv1x1_2-8')
        #步长为2的卷积
        w9=tf.get_variable('w9',[k_h,k_w,output.get_shape()[1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        output=tf.nn.conv2d(output,w9,strides=[1,1,2,2],padding='SAME',data_format='NCHW')
        output=tf.nn.relu(output)
        return output
 #变尺度空洞空间上下文模块+conv  
def Atrous_Conv(name,input_dim,output_dim,filter_size,inputs,k_h=3,k_w=3,he_init=True,biases=True,stddev=0.02):
    with tf.variable_scope(name):  
        #1*1conv 
        output=inputs
        output1=tf.transpose(output,(0,2,3,1))
        
        w10=tf.get_variable('w10',[1,1,output1.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev)) 
        conv1x1=tf.nn.atrous_conv2d(output1,w10,rate=[1,1],padding='SAME',name='conv1x1-9')    
        #3*3空洞卷积
        w11=tf.get_variable('w11',[3,3,output1.get_shape()[-1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv3x3_1=tf.nn.atrous_conv2d(output1,w11,rate=[1,1],padding='SAME',name='conv3x3_1c')
        conv3x3_3=tf.nn.atrous_conv2d(output1,w11,rate=[2,2],padding='SAME',name='conv3x3_3')
        conv3x3_5=tf.nn.atrous_conv2d(output1,w11,rate=[5,5],padding='SAME',name='conv3x3_5')
        
        output2=tf.concat([conv1x1,conv3x3_1,conv3x3_3,conv3x3_5],axis=-1)
        
        #1x1卷积降通道 
        output3=tf.transpose(output2,(0,3,1,2))
        
        output4=slim.conv2d(output3,output_dim,[1,1],stride=1,padding='SAME',data_format='NCHW',scope='conv1x1_3')   
       
        #步长为2的卷积+relu
        w12= tf.get_variable('w12',[k_h,k_w,output4.get_shape()[1],output_dim],initializer=tf.truncated_normal_initializer(stddev=stddev))        
        output5 =tf.nn.conv2d(output4, w12, strides=[1,1,2,2], padding='SAME',data_format='NCHW')
        
        output=tf.nn.relu(output5)
        #卷积conv3*3
        output=slim.conv2d(output,output_dim,filter_size,stride=1,padding='SAME',data_format='NCHW',scope='conv3')
        
        return output    
#变尺度空洞空间上下文残差模块，分为Down、up、None三种模式
def Atrous_and_Res( name,  input_dim, output_dim , filter_size, inputs, resample=None,he_init=True):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        if resample=='down':           
            conv_shortcut = Atrous_Conv
            conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
            conv_2        = functools.partial(Conv_Atrous, input_dim=input_dim, output_dim=output_dim)
        elif resample=='up':
            conv_shortcut = Upsample_Atrous_Conv_1
            conv_1        = functools.partial(Upsample_Atrous_Conv_2, input_dim=input_dim, output_dim=output_dim)
            conv_2        = functools.partial(lib.conv2d.Atrous_Conv2D_3, input_dim=output_dim, output_dim=output_dim)
        elif resample==None:
            conv_shortcut = lib.conv2d.Atrous_Conv2D
            conv_1        = functools.partial(lib.conv2d.Atrous_Conv2D_1, input_dim=input_dim, output_dim=input_dim)
            conv_2        = functools.partial(lib.conv2d.Atrous_Conv2D_2, input_dim=input_dim, output_dim=output_dim)
        else:
            raise Exception('invalid resample value')
          
        if output_dim==input_dim and resample==None:
           
            shortcut = inputs # Identity skip-connection
        else:                      
            shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True,inputs=inputs)      
      # inputs_size=tf.shape(inputs)     
       #image_output=slim.conv2d(inputs, depth, [1,1],data_format='NCHW' ,scope='image_output_conv1x1',activation=None)
       #image_output=tf.image.resize_bilinear(image_output,(inputs_size[1],inputs_size[2]))
        output = inputs
        output = Normalize(name+'.BN1', [0,2,3], output)
        output = tf.nn.relu(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False) 
        
       # if Atrous_add==True:                                
       #     conv1x1=slim.conv2d(output, output_dim,[1,1], data_format='NCHW', scope='conv1x1')
        #    conv3x3_1=slim.conv2d(output, de, [3,3], data_format='NCHW', rate=3, scope='conv3x3_1')
       #     conv3x3_2=slim.conv2d(output, depth,[3,3], data_format='NCHW', rate=5, scope='conv3x3_2')
       #     conv3x3_3=slim.conv2d(output, depth,[3,3], data_format='NCHW', rate=7, scope='conv3x3_3')
            
       #     output=tf.concat([conv1x1,conv3x3_1,conv3x3_2,conv3x3_3],axis=1,name='concat')
        #    output=slim.conv2d(output, depth, [1,1], data_format='NCHW', scope='conv_1x1')           
        output = Normalize(name+'.BN2', [0,2,3], output)
        output = tf.nn.relu(output)
        output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)
       
        return output+shortcut
               
#传统的自注意力机制模块1，reshape实现N*N
def self_attention_1(x,k=8,name='self_attention'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        batch_size, num_channels, height, width = x.get_shape().as_list()

        # 小于k时会出现 num_channels // k = 0,调整k为通道数大小
        if num_channels < k :
            k = num_channels

        x1 =conv1x1(x,output_dim= num_channels//k,name='x1_conv1x1')
        
        x1=tf.reshape(x1,[-1,height*width,num_channels//k])
        x2=tf.transpose(x1,(0,2,1))
        x3=tf.matmul(x1,x2)
        
        x3=tf.nn.softmax(x3)
        
        x_out=tf.matmul(x3,x1)
        
        x_out=tf.reshape(x_out,[-1,num_channels//k,height,width])
        x_out=conv1x1(x_out,output_dim=num_channels,name='conv1x1')
        
        sigma=tf.get_variable('sigma_ratio',[1],initializer=tf.constant_initializer(0.0))
        
        
        result = x + x_out*sigma

        return result

#变尺度自注意力机制模块
def Mutil_self_attention(x, k=8,name='Mutil_self_attention',stddev=0.02,m=None):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        # 小于k时会出现 num_channels // k = 0,调整k为通道数大小

        batch_size,num_channels,height,width=x.get_shape().as_list()
        if num_channels < k:
            k=num_channels
        #变尺度空洞空间上下文模块
        f=tf.transpose(x,(0,2,3,1))
       
        w7q=tf.get_variable('w7q',[1,1,f.get_shape()[-1],num_channels],initializer=tf.truncated_normal_initializer(stddev=stddev)) 
        conv1x1_1=tf.nn.atrous_conv2d(f, w7q, rate=[1,1], padding='SAME',name='conv1x1_')  
       
        w8s=tf.get_variable('w8s',[3,3,f.get_shape()[-1],num_channels],initializer=tf.truncated_normal_initializer(stddev=stddev)) 
        conv3x3_2=tf.nn.atrous_conv2d(f, w8s, rate=[2,2], padding='SAME', name='conv3x3_')
        conv3x3_3=tf.nn.atrous_conv2d(f, w8s, rate=[3,3], padding='SAME', name='conv3x3_')   
       
        f=tf.concat([conv1x1_1,conv3x3_2,conv3x3_3],axis=-1)  
           
        f1=tf.transpose(f,(0,3,1,2))#B,C,H,W
        #------------------------------------------   
        f2=slim.conv2d(f1,num_channels//k,[1,1],stride=1,padding='SAME',data_format='NCHW',scope='f-con1x1')  #N,C',H,W
        print('f2',f2.shape)
           #f =conv1x1(x,output_dim= num_channels//k,name='f_conv1x1')
        f3=tf.reshape(f2,[-1,num_channels//k,height*width])#C',N
        g1=tf.transpose(f3,(0,2,1))#N,C'
        
        s=tf.matmul(g1,f3)#N,N
        
        beta=tf.nn.softmax(s)#N,N
        o=tf.matmul(f3,beta)#C',N
        
        o = tf.reshape(o,shape=[-1, num_channels, height, width])#C,H,W
        print('o',o.shape)
            
        sigma = tf.get_variable("sigma_ratio1", [1], initializer=tf.constant_initializer(0.0))           
        result = x + o*sigma
        return result      
#传统的自注意力机制模块2，flatten实现N*N   
def self_attention(x,k=8,name='self_attention'):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        batch_size, num_channels, height, width = x.get_shape().as_list()

        # 小于k时会出现 num_channels // k = 0,调整k为通道数大小
        if num_channels < k :
            k = num_channels

        f =conv1x1(x,output_dim= num_channels//k,name='f_conv1x1')
        g =conv1x1(x,output_dim= num_channels//k,name='h_conv1x1')
        h =conv1x1(x,output_dim= num_channels,name='g_conv1x1')
        
        # 将f，g，h展开，方便下面的矩阵相乘
        flatten_f, flatten_g, flatten_h = tf.layers.flatten(f) ,tf.layers.flatten(g),tf.layers.flatten(h)
        
        s = tf.matmul(flatten_g,flatten_f,transpose_b=True)

        # attention map
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta,flatten_h)
        o = tf.reshape(o,shape=[-1, num_channels, height, width])

        sigma = tf.get_variable("sigma_ratio", [1], initializer=tf.constant_initializer(0.0))
        result = x + sigma*o

        return result
#加入两层自注意力机制的残差模块，有down、up、none三种模式
def SA_Res_SA(name, input_dim, output_dim, filter_size, inputs,sa=True, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.conv2d.Conv2D
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)

    # 增加一层自注意力机制
    if sa==True:
        output = self_attention(output,name=name+'sa1')
        #output=Mutil_self_attention(output,name=name+'sa1')
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)
    # 增加一层自注意力机制
    if sa==True:
        output = self_attention(output,name=name+'sa2')
        #output=Mutil_self_attention(output,name=name+'sa2')
    return shortcut + output

#用于搭建多维度特征融合自注意力机制残差模块
def ResidualBlock_mutil(name, input_dim, output_dim, filter_size, inputs,input_add, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = MeanPoolConv
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = UpsampleConv
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.conv2d.Conv2D
        conv_1        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
        conv_2        = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:

        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)
        
    output = inputs
    output = Normalize(name+'.BN1', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
    
    output = Normalize(name+'.BN2', [0,2,3], output)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)
    
    input_add=slim.conv2d(input_add,output_dim,[3,3],stride=1,padding='SAME',data_format='NCHW')
    output=output+input_add
    return shortcut + output

#多维度特征融合自注意力机制残差模块1
def Mutil_feature_1(name,input_low,input_height,input3,input_dim,output_dim,output_dim1,output_dim2,resample=None,stddev=0.02):
    with tf.variable_scope(name): 
        w_d=tf.get_variable('w_d',[3,3,input_low.get_shape()[1],output_dim1],initializer=tf.truncated_normal_initializer(stddev=stddev)) 
        x_low=self_attention_1(input_low)
        x_low=tf.nn.conv2d(x_low, w_d, strides=[1,1,1,1], padding='SAME',data_format='NCHW')#低维度特征
        
        w_r=tf.get_variable('w_r',[3,3,input_height.get_shape()[1],output_dim2],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_height=tf.nn.conv2d(input_height,w_r,strides=[1,1,1,1],padding='SAME',data_format='NCHW')#高纬度特征
        
        output=tf.multiply(x_low,x_height,name='mul_1')#高低维度特征融合
        print('output_and',output.shape)#看一下是不是要改变通道数
        
        w_p=tf.get_variable('w_p',[3,3,output.get_shape()[1],x_low.get_shape()[1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_low_1=tf.nn.conv2d(output,w_p,strides=[1,1,1,1],padding='SAME',data_format='NCHW')#
        print('x_low_1',x_low_1.shape)
        
        w_e=tf.get_variable('w_e',[3,3,output.get_shape()[1],x_height.get_shape()[1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_height_1=tf.nn.conv2d(output,w_e,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
        print('x_height_1',x_height_1.shape)
        
        output_low=x_low+x_low_1#低维度特征优化
        output_height=x_height+x_height_1#高纬度特征优化
        output_and=output_low+output_height
        output=ResidualBlock_mutil('RES_Mutil1',input_dim,output_dim,3,input3,output_and,resample=None)#残差连接
        return output
#多维度特征融合自注意力机制残差模块2
def Mutil_feature_2(name,input_low,input_height,input3,input_dim,output_dim,output_dim1,output_dim2,resample=None,stddev=0.02):
    with tf.variable_scope(name): 
        w_d=tf.get_variable('w_d',[3,3,input_low.get_shape()[1],output_dim1],initializer=tf.truncated_normal_initializer(stddev=stddev)) 
        x_low=self_attention_1(input_low)
        x_low=tf.nn.conv2d(x_low, w_d, strides=[1,1,1,1], padding='SAME',data_format='NCHW')#低维度特征
        
        w_r=tf.get_variable('w_r',[3,3,input_height.get_shape()[1],output_dim2],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_height=tf.nn.conv2d(input_height,w_r,strides=[1,1,1,1],padding='SAME',data_format='NCHW')#高维度特征
        
        output=tf.multiply(x_low,x_height,name='mul_1')#高、低维度特征融合，为细化特征准备
        print('output_and',output.shape)#看一下是不是要改变通道数
         
        w_p=tf.get_variable('w_p',[3,3,output.get_shape()[1],x_low.get_shape()[1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_low_1=tf.nn.conv2d(output,w_p,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
        print('x_low_1',x_low_1.shape)
        
        w_e=tf.get_variable('w_e',[3,3,output.get_shape()[1],x_height.get_shape()[1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_height_1=tf.nn.conv2d(output,w_e,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
        print('x_height_1',x_height_1.shape)
        
        output_low=x_low+x_low_1
        output_height=x_height+x_height_1#高、低维度特征优化
        output_and=output_low+output_height
        output=ResidualBlock_mutil('RES_Mutil2',input_dim,output_dim,3,input3,output_and,resample=None)
        return output
#多维度特征融合自注意力机制残差模块3
def Mutil_feature_3(name,input_low,input_height,input3,input_dim,output_dim,output_dim1,output_dim2,resample=None,stddev=0.02):
    with tf.variable_scope(name): 
        w_d=tf.get_variable('w_d',[3,3,input_low.get_shape()[1],output_dim1],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_low=self_attention(input_low)
        x_low=tf.nn.conv2d(x_low, w_d, strides=[1,1,1,1], padding='SAME',data_format='NCHW')
        
        w_r=tf.get_variable('w_r',[3,3,input_height.get_shape()[1],output_dim2],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_height=tf.nn.conv2d(input_height,w_r,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
        
        output=tf.multiply(x_low,x_height,name='mul_1')
        print('output_mut',output.shape)#看一下是不是要改变通道数
         
        w_p=tf.get_variable('w_p',[3,3,output.get_shape()[1],x_low.get_shape()[1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_low_1=tf.nn.conv2d(output,w_p,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
        print('x_low_1',x_low_1.shape)
        
        w_e=tf.get_variable('w_e',[3,3,output.get_shape()[1],x_height.get_shape()[1]],initializer=tf.truncated_normal_initializer(stddev=stddev))
        x_height_1=tf.nn.conv2d(output,w_e,strides=[1,1,1,1],padding='SAME',data_format='NCHW')
        print('x_height_1',x_height_1.shape)
        
        output_low=x_low+x_low_1
        print('output_low',output_low.shape)
        output_height=x_height+x_height_1
        print('output_height',output_height.shape)
        output_and=output_low+output_height
        output=ResidualBlock_mutil('RES_Mutil3',input_dim,output_dim,3,input3,output_and,resample=None)
        return output
# ! Detectors
#检测网络，encoder+decoder两部分，输入有inputs和input2，输出gen_img和x,其中x为encoder的输出，gen_img为decoder的输出
def Detector( inputs,  m=None, SA=None, alpha=0.6, input2=None, fuse=None, dim=8, nonlinearity=tf.nn.relu):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Detector')]) > 0
    with tf.variable_scope('Detector', reuse=reuse):
        print('------------Detector------------\n inputs shape :',inputs.shape)
        if SA==0:#传统的编解码结构，残差块组成        
            output1 = lib.conv2d.Conv2D('Detector.Input', 3, dim, 3, inputs, he_init=False)#8,192,256
            output1 = ResidualBlock('Detector.Res1', dim, 2*dim, 3, output1, resample='down')#16,96,128
            output1 = ResidualBlock('Detector.Res2', 2*dim, 4*dim, 3, output1, resample='down') #32,48,64          
            output1 = ResidualBlock('Detector.Res3', 4*dim, 8*dim, 3, output1, resample='down') #64，24,32      
                  
            output1 = ResidualBlock('Detector.Res4a', 8*dim, 8*dim, 3, output1, resample=None)#64,24,32            
            output2 = ResidualBlock('Detector.Res4b', 8*dim, 8*dim, 3, output1, resample=None)#64,24,32 
            x=tf.tanh(output2)#获得特征图feature_map
            if m==1:
                x=(1.0-alpha)*x+alpha*input2#获得混合特征图，combine_feature,64,24,32
               
            output2 = ResidualBlock('Detector.Res5', 8*dim, 8*dim, 3, x, resample='up')  #64, 48,64        
            output2 = ResidualBlock('Detector.Res6', 8*dim, 4*dim, 3, output2, resample='up') #32,96,128           
            output2 = ResidualBlock('Detector.Res7', 4*dim, 2*dim, 3, output2, resample='up') #16,192,256                       
            output2 = ResidualBlock('Detector.Res8', 2*dim, 1*dim, 3, output2, resample=None)#8,192,256
            
            output2 = tf.nn.relu(output2)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage', 1*dim, 3, 3, output2)#3,192,256
            gen_img = tf.nn.relu(gen_img)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage2', 3, 1, 3, gen_img)#1,192,256
            gen_img = tf.nn.relu(gen_img)
            print('output shape',gen_img.shape)
           # output2 = Normalize('Detector.Output', [0,2,3], output2)
           # output2 = tf.nn.relu(output2)
           # gen_img = lib.ops.conv2d.Conv2D('Detector.Toimage', 1*dim, 1, 3, output2)
           # gen_img = tf.sigmoid(gen_img)            
            return  gen_img, x
            
        elif SA==1:#网络结构a，变尺度空洞上下文模块，步长为2的卷积代替所有的池化层     
            output1 = lib.conv2d.Conv2D('Detector.Input', 3, dim, 3, inputs, he_init=False)#8,192,256
            print('output1',output1.shape)
            output1 = Atrous_and_Res('Detector.Atrous_and_Res1', dim, 2*dim, 3, output1,  resample='down')#16,96,128
            print('output2',output1.shape)
            output1 = Atrous_and_Res('Detector.Atrous_and_Res2', 2*dim, 4*dim, 3, output1, resample='down')#32,48,64
            print('output3',output1.shape)
            output1 = Atrous_and_Res('Detector.Atrous_and_Res3', 4*dim, 8*dim, 3, output1, resample='down')#64,24,32
            print('output4',output1.shape)
            #output1 = SA_ResidualBlock('Detector.Res4', 8*dim, 8*dim, 3, output1, sa=True,resample='down')
                      
            output1=Atrous_and_Res('Detector.Atrous_and_Res4a', 8*dim, 8*dim,  3, output1, resample=None)#64,24,32
            print('output5',output1.shape)
            output2=Atrous_and_Res('Detector.Atrous_and_Res4b',8*dim, 8*dim,  3, output1,  resample=None)#64,24,32  
            print('output6',output2.shape)
            x=tf.tanh(output2)#获得特征图feature_map
            if m==1:
                x=(1.0-alpha)*x+alpha*input2#获得混合特征图，combine_feature,64,24,32
               
            output2 =Atrous_and_Res('Detector.Atrous_and_Res5', 8*dim, 8*dim, 3, x, resample='up')#64,48,64
            print('output7',output2.shape)
            output2 = Atrous_and_Res('Detector.Atrous_and_Res6', 8*dim, 4*dim, 3, output2, resample='up')#32,96,128
            print('output8',output2.shape)
            output2 = Atrous_and_Res('Detector.Res7', 4*dim, 2*dim, 3, output2, resample='up')#16,192,256
            print('output_sa.shape',output2.shape)
            output2 =Atrous_and_Res('Detector.Res8', 2*dim, 1*dim, 3, output2, resample=None)#8,192,256
            print('output10',output2.shape)
            output2 = tf.nn.relu(output2)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage', 1*dim, 3, 3, output2)#3,192,256
            gen_img = tf.nn.relu(gen_img)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage2', 3, 1, 3, gen_img)#1,192,256
            gen_img = tf.nn.relu(gen_img)
            print('output shape',gen_img.shape)
            
            return  gen_img, x
        
        elif SA==2:  #网络结构b,跨层的跳跃连接穿过由残差连接而成的变尺度空洞空间上下文模块，步长为2的卷积代替部分池化层          
            output1 = lib.conv2d.Conv2D('Detector.Input', 3, dim, 3, inputs, he_init=False)#8,192,256
            print('output1',output1.shape)
            
            output1 = Res_conv_d2('Detector.Res1', dim, 2*dim, 3, output1, resample='down')#16,96,128
            print('output1 shape',output1.shape)
            x1=output1#16,96,128
            output1 = ResidualBlock('Detector.Res2', 2*dim, 4*dim, 3, output1, resample='down') #32,48,64 
            print('output1 shape1',output1.shape)
            x2=output1#32,48,64
            output1 = Res_conv_d2('Detector.Res3', 4*dim, 8*dim, 3, output1, resample='down')#64,24,32
            x3=output1#64,24,32
            print('x3',x3.shape)
            output1=Atrous_and_Res('Detector.Atrous_and_Res4a', 8*dim, 8*dim,  3, output1, resample=None)#64,24,32
            output2=Atrous_and_Res('Detector.Atrous_and_Res4b',8*dim, 8*dim,  3, output1,  resample=None)#64,24,32             
            x=tf.tanh(output2)#获得特征图feature_map
            if m==1:
                x=(1.0-alpha)*x+alpha*input2#获得混合特征图，combine_feature,64,24,32
 
            x4=Atrous_Conv_res('root1',x3, 8*dim, 8*dim, 3)#64,24,32
            print('x4',x4.shape)
            x5=x4+x
            output2 = Atrous_and_Res('Detector.Res5', 8*dim, 8*dim, 3, x5, resample='up')  #64, 48,64    
            print('out2',output2.shape)#64,48,64
            x6=Atrous_Conv_res('root2', x2, 4*dim, 8*dim, 3)#64,48,64
            print('x6',x6.shape)
            x7=x6+output2
            output2 = Atrous_and_Res('Detector.Res6', 8*dim, 4*dim, 3, x7, resample='up')  #32, 96,128
            x8=Atrous_Conv_res('root3', x1 , 2*dim, 4*dim, 3)#32,96,128
            print('x8',x8.shape)
            x9=x8+output2
            output2 = Atrous_and_Res('Detector.Res7', 4*dim, 2*dim, 3, x9, resample='up')  #16, 192,256
            
            output2 = ResidualBlock('Detector.Res8', 2*dim, 1*dim, 3, output2, resample=None)#8,192,256

            output2 = tf.nn.relu(output2)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage', 1*dim, 3, 3, output2)
            gen_img = tf.nn.relu(gen_img)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage2', 3, 1, 3, gen_img)
            gen_img = tf.nn.relu(gen_img)
            print('output shape',gen_img.shape)           
            #output2 = Normalize('Detector.Output', [0,2,3], output2)
           # output2 = tf.nn.relu(output2)
          #  gen_img = lib.ops.conv2d.Conv2D('Detector.Toimage', 1*dim, 1, 3, output2)
           #gen_img = tf.sigmoid(gen_img)           
            return  gen_img, x      
        elif SA==3:#网络结构c，跨层的跳跃连接穿过变尺度空洞空间上下文模块+传统的或变尺度自注意力机制模块
            output1 = lib.conv2d.Conv2D('Detector.Input', 3, dim, 3, inputs, he_init=False)#8,192,256
            output1 = SA_Res_SA('Detector.Res1', dim, 2*dim, 3, output1, sa=True, resample='down')#16,96,128
            x1=Atrous_Conv_res('root1',output1, 2*dim, 4*dim, 3)#32,96,128
            
            output1 = SA_Res_SA('Detector.Res2', 2*dim, 4*dim, 3, output1, sa=True, resample='down') #32,48,64  
            x2=Atrous_Conv_res('root2',output1, 4*dim, 8*dim, 3)#64,48,64
            
            output1 = SA_Res_SA('Detector.Res3', 4*dim, 8*dim, 3, output1, sa=True, resample='down') #64，24,32      
            x3=Atrous_Conv_res('root3',output1, 8*dim, 8*dim, 3) #64,24,32    
            
            output1 = SA_Res_SA('Detector.Res4a', 8*dim, 8*dim, 3, output1,sa=True, resample=None)#64,24,32            
            output2 = SA_Res_SA('Detector.Res4b', 8*dim, 8*dim, 3, output1, sa=True,resample=None)#64,24,32 
            x=tf.tanh(output2)#获得特征图feature_map
            if m==1:
                x=(1.0-alpha)*x+alpha*input2#获得混合特征图，combine_feature,64,24,32
            x4=x+x3   
            output2 = SA_Res_SA('Detector.Res5', 8*dim, 8*dim, 3, x4, sa=True, resample='up')  #64, 48,64   
            
            x5=output2+x2
            output2 = SA_Res_SA('Detector.Res6', 8*dim, 4*dim, 3, x5, sa=True, resample='up') #32,96,128    
            
            x6=output2+x1
            output2 = SA_Res_SA('Detector.Res7', 4*dim, 2*dim, 3, x6, sa=True, resample='up') #16,192,256   
                    
            output2 = SA_Res_SA('Detector.Res8', 2*dim, 1*dim, 3, output2, sa=True, resample=None)#8,192,256
            
            output2 = tf.nn.relu(output2)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage', 1*dim, 3, 3, output2)#3,192,256
            gen_img = tf.nn.relu(gen_img)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage2', 3, 1, 3, gen_img)#1,192,256
            gen_img = tf.nn.relu(gen_img)
            print('output shape',gen_img.shape)
           # output2 = Normalize('Detector.Output', [0,2,3], output2)
           # output2 = tf.nn.relu(output2)
           # gen_img = lib.ops.conv2d.Conv2D('Detector.Toimage', 1*dim, 1, 3, output2)
           # gen_img = tf.sigmoid(gen_img)            
            return  gen_img, x
        elif SA==4:#添加传统的或变尺度自注意力机制的网络结构，因频繁使用变尺度自注意力机制模块导致计算量过大，所以受限于显卡性能
            output1 = lib.conv2d.Conv2D('Detector.Input', 3, dim, 3, inputs, he_init=False)#8,192,256
            output1 = SA_Res_SA('Detector.Res1', dim, 2*dim, 3, output1, sa=True, resample='down')#16,96,128
            x1=lib.conv2d.Conv2D('conv1', 2*dim, 4*dim, 3, output1, he_init=False)#32,96,128
            
            output1 = SA_Res_SA('Detector.Res2', 2*dim, 4*dim, 3, output1, sa=True, resample='down') #32,48,64 
            x2=lib.conv2d.Conv2D('conv2',4*dim, 8*dim, 3, output1, he_init=False)#64,48,64
            
            output1 = SA_Res_SA('Detector.Res3', 4*dim, 8*dim, 3,output1, sa=True, resample='down') #64，24,32      
            x3=lib.conv2d.Conv2D('conv3', 8*dim,8*dim, 3, output1, he_init=False)#64,24,32   
            
            output1 = SA_Res_SA('Detector.Res4a', 8*dim, 8*dim, 3, output1, sa=True,resample=None)#64,24,32 
            
            output2 = SA_Res_SA('Detector.Res4b', 8*dim, 8*dim, 3, x,sa=True,resample=None)#64,24,32 
            x=tf.tanh(output2)#获得特征图feature_map
            if m==1:
                x=(1.0-alpha)*x+alpha*input2#获得混合特征图，combine_feature,64,24,32
            x4=x+x3   
            output2 = SA_Res_SA('Detector.Res5', 8*dim, 8*dim, 3, x4, sa=True,resample='up')  #64, 48,64  
            x5=output2+x2
            output2 = SA_Res_SA('Detector.Res6', 8*dim, 4*dim, 3, x5, sa=True, resample='up') #32,96,128   
            x6=output2+x1
            output2 = SA_Res_SA('Detector.Res7', 4*dim, 2*dim, 3, x6,sa=True, resample='up') #16,192,256  
           
            output2 = SA_Res_SA('Detector.Res8', 2*dim, 1*dim, 3,output2, sa=True, resample=None)#8,192,256
            
            output2 = tf.nn.relu(output2)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage', 1*dim, 3, 3, output2)#3,192,256
            gen_img = tf.nn.relu(gen_img)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage2', 3, 1, 3, gen_img)#1,192,256
            gen_img = tf.nn.relu(gen_img)
            print('output shape',gen_img.shape)
           # output2 = Normalize('Detector.Output', [0,2,3], output2)
           # output2 = tf.nn.relu(output2)
           # gen_img = lib.ops.conv2d.Conv2D('Detector.Toimage', 1*dim, 1, 3, output2)
           # gen_img = tf.sigmoid(gen_img)            
            return  gen_img, x
        
        elif SA==5:#网络结构d，变尺度空洞空间上下文模块和多维度特征融合自注意力机制残差模块的结合
            output1 = lib.conv2d.Conv2D('Detector.Input', 3, dim, 3, inputs, he_init=False)#8,192,256
            x_low_1 = output1#8,192,256
           
            output1 = Res_conv_d2('Detector.Res1', dim, 2*dim, 3, output1, resample='down')#16,96,128
            x_height_1=output1#16,96,128
            
            x_low_2 = output1#16,96,128
            output1 = Res_conv_d2('Detector.Res2', 2*dim, 4*dim, 3, output1, resample='down') #32,48,64   
            x_height_2=output1#32,48,64
            
            x_low_3 = output1#32,48,64
            output1 = Res_conv_d2('Detector.Res3', 4*dim, 8*dim, 3, output1, resample='down') #64，24,32   
            x_height_3=output1#64,24,32
            
            output1 = ResidualBlock('Detector.Res4a', 8*dim, 8*dim, 3, output1, resample=None)#64,24,32            
            output2 = ResidualBlock('Detector.Res4b', 8*dim, 8*dim, 3, output1, resample=None)#64,24,32 
            x=tf.tanh(output2)#获得特征图feature_map
            if m==1:
                x=(1.0-alpha)*x+alpha*input2#获得混合特征图，combine_feature,64,24,32
            #特征融合1
            x_low_3 = Atrous_and_Res('Detector.Atrous_and_Res5', 4*dim, 4*dim, 3, x_low_3, resample='down')#32,24,32   
            print('x_low_3',x_low_3.shape)
            x_height_3=Atrous_Conv_res('root1',x_height_3, 8*dim, 8*dim, 3) #64,24,32  
            print('x_height_3',x_height_3.shape)
            x3= Mutil_feature_1('Mutil.feature.1',x_low_3,x_height_3,x, 8*dim, 8*dim, 8*dim, 8*dim,resample='None')#64,24,32
            print('x3',x3.shape)
            if fuse==True:
                fuse1=conv1x1(x3, output_dim=1, name='fuse1')#1,24,32
                fuse1=ResidualBlock('Detector.fuse1',1,1,3,fuse1,resample='up')#1,48,64
                fuse1=ResidualBlock('Detector.fuse1_1',1,1,3,fuse1,resample='up')#1,96,128
                fuse1=ResidualBlock('Detector.fuse1_2',1,1,3,fuse1,resample='up')#1,192,256
                fuse1=tf.sigmoid(fuse1)
            output2 = ResidualBlock('Detector.Res5', 8*dim, 8*dim, 3, x3, resample='up')  #64, 48,64    
            print('output2',output2.shape)
            #特征融合2
            x_low_2= Atrous_and_Res('Detector.Atrous_and_Res6', 2*dim, 2*dim, 3, x_low_2, resample='down')#16,48,64  
            print('x_low_2',x_low_2.shape)
            x_height_2=Atrous_Conv_res('root2',x_height_2, 4*dim, 4*dim, 3)#32,48,64
            print('x_height_2',x_height_2.shape)
            x2=Mutil_feature_2('Mutil.feature.2',x_low_2,x_height_2, output2, 8*dim, 8*dim, 4*dim, 4*dim, resample='None')#64,48,64
            print('x',x2.shape)
            if fuse==True:
                fuse2=conv1x1(x2,output_dim=1,name='fuse2')#1,48,64       
                fuse2=ResidualBlock('Detector.fusse2',1,1,3,fuse2,resample='up')#1,96,128
                fuse2=ResidualBlock('Detector.fuse2_1',1,1,3,fuse2,resample='up')#,1,192,256
                fuse2=tf.sigmoid(fuse2)
            output2 = ResidualBlock('Detector.Res6', 8*dim, 4*dim, 3, x2, resample='up') #32,96,128    
            print('output2',output2.shape)
            #特征融合3
            x_low_1=Atrous_and_Res('Detector.Atrous_and_Res7', dim, dim, 3, x_low_1, resample='down')#8,96,128
            print('x_low_1',x_low_1.shape)
            x_height_1=Atrous_Conv_res('root3',x_height_1, 2*dim, 2*dim, 3)#16,96,128
            print('x_height_1',x_height_1.shape)
            x1=Mutil_feature_3('Mutil.feature.3', x_low_1, x_height_1, output2, 4*dim, 4*dim, 2*dim, 2*dim, resample='None')#32,96,128
            print('x',x1.shape)
            if fuse==True:
                fuse3=conv1x1(x1,output_dim=1,name='fuse3')#1,96,128               
                fuse3=ResidualBlock('Detector.fuse3',1,1,3,fuse3,resample='up')#1,192,256
                fuse3=tf.sigmoid(fuse3)
            output2 = ResidualBlock('Detector.Res7', 4*dim, 2*dim, 3, x1, resample='up') #16,192,256   
            print('output2',output2.shape)
                    
            output2 = ResidualBlock('Detector.Res8', 2*dim, 1*dim, 3, output2, resample=None)#8,192,256
            
            output2 = tf.nn.relu(output2)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage', 1*dim, 3, 3, output2)#3,192,256
            gen_img = tf.nn.relu(gen_img)
            gen_img = lib.conv2d.Conv2D('Detector.Toimage2', 3, 1, 3, gen_img)#1,192,256
            gen_img = tf.nn.relu(gen_img)
            print('output shape',gen_img.shape)
           # output2 = Normalize('Detector.Output', [0,2,3], output2)
           # output2 = tf.nn.relu(output2)
           # gen_img = lib.ops.conv2d.Conv2D('Detector.Toimage', 1*dim, 1, 3, output2)
           # gen_img = tf.sigmoid(gen_img)            
            return  gen_img, x,fuse1,fuse2,fuse3
        else:
             raise Exception('invalid SA value')
                                    
#伪噪声生成器网络encoder2
def encoder2(inputs,  dim=8, ):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('encoder2')]) > 0
    with tf.variable_scope('encoder2', reuse=reuse):
        print('------------encoder2------------\n inputs hsape :',inputs.shape)

        output = lib.conv2d.Conv2D('encoder2.Input', 4, dim, 3, inputs,    he_init=False)#8,192,256
        output = ResidualBlock('encoder2.Res1', dim, 2*dim, 3, output,  resample='down')#16,96,128
        output = ResidualBlock('encoder2.Res2', 2*dim, 4*dim, 3, output,  resample='down')#32,48,64
        output = ResidualBlock('encoder2.Res3', 4*dim, 8*dim, 3, output, resample='down')#64,24,32
        output = ResidualBlock('encoder2.Res4', 8*dim, 8*dim, 3, output, resample=None)#64,24,32

        output = tf.sigmoid(output)
        print('output shape',output.shape)
        return  output

#鉴别器网络Discriminator
def Discriminator(inputs, dim=8 ,SA_D=None):
    reuse = len([t for t in tf.global_variables() if t.name.startswith('Discriminator')]) > 0
    with tf.variable_scope('Discriminator', reuse=reuse):
        print('------------Discriminator------------\n inputs shappe :',inputs.shape)
        if SA_D==0: #传统的残差结构
            
            output = ResidualBlock('Discriminator.Res1', 8*dim, 8*dim, 3, inputs, resample=None)
            output = ResidualBlock('Discriminator.Res2', 8*dim, 8*dim, 3, output, resample=None)
            output = ResidualBlock('Discriminator.Res3', 8*dim, 4*dim, 3, output, resample=None)
            output = ResidualBlock('Discriminator.Res4', 4*dim, 2*dim, 3, output, resample=None)   
            disc_img = ResidualBlock('Discriminator.Res5', 2*dim, 1*dim, 3, output, resample=None)   
            
                      
            disc_result= tf.reduce_mean(disc_img , axis=[1,2,3])
            print('output shape',disc_result.shape)
        
        elif SA_D==1:#变尺度空洞空间上下文模块
            output1 = Atrous_and_Res('Disc.Res1', 8*dim,  8*dim, 3, inputs, resample=None)
            print('ouput-d1',output1.shape)
            output1 = Atrous_and_Res('Disc.Res2', 8*dim, 8*dim,3, output1, resample=None)
            print('output-d2',output1.shape)
            output1 = Atrous_and_Res('Disc.Res3', 8*dim, 4*dim,3, output1,resample=None)
            output1 = Atrous_and_Res('Disc.Res4', 4*dim, 2*dim, 3, output1, resample=None)
            disc_img = Atrous_and_Res('Disc.Res5', 2*dim, 1*dim, 3, output1, resample=None)   
            print('disc_img shape',disc_img.shape)

            disc_result = tf.reduce_mean(disc_img, axis=[1,2,3],name='Discriminator.resault')
            print('output shape',disc_result.shape)     
       
        elif SA_D==2:#传统的或变尺度自注意力机制
          
            output =SA_Res_SA('Discriminator.SA_Res1', 8*dim, 8*dim, 3, inputs,  sa=True ,resample=None)
            output = SA_Res_SA('Discriminator.SA_Res2', 8*dim, 8*dim, 3, output,  sa=True ,resample=None)
            output = SA_Res_SA('Discriminator.SA_Res3', 8*dim, 4*dim, 3, output, sa=True , resample=None)
            output = SA_Res_SA('Discriminator.SA_Res4', 4*dim, 2*dim, 3, output,  sa=True ,resample=None)
            disc_img = SA_Res_SA('Discriminator.SA_Res5', 2*dim, 1*dim, 3, output, sa=True, resample=None)
            print('disc_img shape',disc_img.shape)

            disc_result = tf.reduce_mean(disc_img, axis=[1,2,3],name='Discriminator.resault')
            print('output shape',disc_result.shape)
                                       
        else:
             raise Exception('invalid SA value')
            
        return disc_result
