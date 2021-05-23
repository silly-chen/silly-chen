import tensorflow as tf
import reader,  cv2,os,time,sys
import numpy as np
import model
import ops
from tflib import conv2d
import tflib as lib


tf.reset_default_graph()#用于清除默认图像堆栈并重置全局默认图形，该函数只适用于当前线程，如果没有它，会在上一次执行的基础上生成新张量

epoch =2000    #总迭代次数
batch_size =16#批次大小
h,w = 192, 256    # 需要输入网络的图片大小，会在读取的时候resize
a = 0.6    # 超参数,噪声所占比例，可设置0.1~0.9
train_data_rate = 0.8    # 训练集所占比例，划分训练集和测试集

SA=5   #检测器网络结构选择变量              
          
SA_D=1  #鉴别器网络结构选择变量           

data_dir = './dataset_new/image'    #选择数据库
#data_dir = './dataset_mini/image'    # 小样本测试程序用，共12张图片

data_enhance = None  #是否使用数据增强，为1时进行数剧增强
log_dir = 'log/train.log'   # 设置日志保存路径


 # 初始化log
train_log = ops.Logger(log_dir,level='debug')

# 创建记录保存的图片文件夹
if not os.path.exists('./data_save/record'):
    os.makedirs('./data_save/record')
if not os.path.exists('./data_save/final/train/'):
    os.makedirs('./data_save/final/train/')
if not os.path.exists('./data_save/final/test/'):
    os.makedirs('./data_save/final/test/')

# 保存测试数据
if not os.path.exists('./data_save/test'):
    os.makedirs('./data_save/test')

# 定义占位符
image = tf.placeholder(tf.float32, [None, 3,h, w,])#CHW                   
noise_image = tf.placeholder(tf.float32, [None,4,h, w])
# 单通道label
label = tf.placeholder(tf.float32, [None,1, h, w])



#获取检测器网络输出,x3,x2,x1是纯融合特征
real_lab,real_feature,x3,x2,x1=model.Detector(inputs=image,SA=SA,fuse=True)


#获取伪噪声发生器网络encoder2的输出
fake_feature = model.encoder2(noise_image)
#fake_feature=model.Disturber(noise_image)

# 特征叠加，x4,x5,x6是混合噪声的融合特征
com_lab,com_feature,x4,x5,x6=model.Detector(inputs=image,m=1,SA=SA,input2=fake_feature,fuse=True)
#combine_feature = (1.0-a)*real_feature + a*fake_feature


# 真特征鉴别器输出
disc_real = model.Discriminator(inputs=real_feature,SA_D=SA_D)
# 混合特征鉴别器输出
disc_com = model.Discriminator(inputs=com_feature,SA_D=SA_D)
#各类交叉熵损失函数
def cross_entropy(logits,labels,m):
    if m==0:
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))#sigmoid交叉熵
        print('loss',loss.shape)
    elif m==1:
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels))#softmax交叉熵,损失值会到一百多这样
    elif m==2:
        loss=tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=logits,targets=labels,pos_weight=2.0))#带权重的交叉熵,时间相对会长一点
    else:
        raise Exception('invalid m value')
    return loss
#各类回归损失函数
def Regression_loss(y, y_pred , m):
    if m==0:
        loss=tf.reduce_mean(tf.abs(y-(y_pred)))/(tf.reduce_mean(tf.abs(y+y_pred))+0.0001)#绝对误差损失
    elif m==1:
        loss=tf.reduce_mean(tf.square(y-(y_pred)))/(tf.reduce_mean(tf.square(y+y_pred))+0.0001)#均方差损失
    else:
        raise Exception('invalid m value')
    return loss
#Huber损失
def Huber_loss(labels,pred,delta=1):
    error=labels-pred
    error1=labels+pred
    is_small_error=(tf.reduce_mean(tf.abs(error))/(tf.reduce_mean(tf.abs(error1))+0.0001))<delta
    loss=Regression_loss(labels,pred,m=1)
    linear_loss=(delta*(tf.reduce_mean(tf.abs(error))-0.5*delta))/(delta*(tf.reduce_mean(tf.abs(error1))-0.5*delta))
    return tf.where(is_small_error,loss,linear_loss)

def loss_fuse(x1,x2,x3,label):     
    loss_x1=cross_entropy(logits=x1,labels=tf.sigmoid(label),m=0)
    loss_x2=cross_entropy(logits=x2,labels=tf.sigmoid(label),m=0)
    loss_x3=cross_entropy(logits=x3,labels=tf.sigmoid(label),m=0)
    loss_fuse=loss_x1+loss_x2+loss_x3
     
    return loss_fuse
#检测器网络的损失函数
def loss_detector(y, y_pred, mutil1,mutil2,mutil3,y_real,y_comb,beta,a):#变量a可以选择不同的检测器网络损失函数
    if a==0:
        loss_detector=Regression_loss(y,y_pred,m=0)#绝对误差损失
    elif a==1:
        loss_detector=Regression_loss(y, y_pred, m=1)#均方差损失
    elif a==2:
        loss_detector=Huber_loss(y,y_pred)#huber损失
    elif a==3:
        loss_detector=cross_entropy(logits=y_real,labels=tf.sigmoid(y),m=0)+cross_entropy(logits=y_comb,labels=tf.sigmoid(y),m=0)+\
                        beta*loss_fuse(mutil1,mutil2,mutil3,y)
    else:
        raise Exception('invalid a value')
        
    return loss_detector   
#鉴别器网络的损失  
def loss_discriminator(y_real, y_fake, b, m ):#变量b可以选择不同的鉴别器损失函数，变量m可以选择不同的交叉熵损失
    if b==0:
        loss_discriminator=cross_entropy(logits=y_real,labels=tf.ones_like(y_real),m=m)+cross_entropy(logits=y_fake,labels=tf.zeros_like(y_fake),m=m)
    elif b==1:
        loss_discriminator=0.5*(tf.reduce_mean(tf.square(y_real-1))+tf.reduce_mean(tf.square(y_fake-0)))
    elif b==2:
        loss_discriminator=0.5*(tf.reduce_mean(tf.square(y_real-1))+tf.reduce_mean(tf.square(y_fake+1)))      
    else:
        raise Exception('invalid b value')
    return loss_discriminator
#伪噪声生成器网络的损失
def loss_encoder( y_fake, y, y_pred, c, m):#变量c可以选择不同的损失函数，变量m可以选择不同的交叉熵函数
    if c==0:
        loss_encoder=cross_entropy(logits=y_fake,labels=tf.ones_like(y_fake),m=m)+Regression_loss(y,1-y_pred,m=0)
    elif c==1:
        loss_encoder=Regression_loss(y,y_pred,m=1)+0.5*tf.reduce_mean(tf.square(y_fake-1))                    
    elif c==2:
        loss_encoder=Regression_loss(y,1-y_pred,m=0)+0.5*tf.reduce_mean(tf.square(y_fake-0))            
    else:
        raise Exception('invalid c value')
    return loss_encoder    

# 鉴别器的损失，b=0/1/2,选择损失函数，当b=0时，可以根据m=0/1/2,选择不同的交叉熵函数
loss_D=loss_discriminator(disc_real, disc_com, b=0,m=0)


# encoder2的损失函数，c=0/1/2,不同的损失函数，当C=0时，可以根据m=0/1/2,选择交叉熵
loss_encoder2=loss_encoder(disc_com,y=label,y_pred=com_lab,c=0,m=0)
#loss_encoder2=loss_encoder(disc_com,y=label,y_pred=com_lab,c=0,m=0)
#loss_encoder2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_com, labels=tf.ones_like(disc_com)))+\
#                                    tf.reduce_mean(tf.abs(com_lab-(1.0-label)))/(tf.reduce_mean(tf.abs(com_lab+1.0-label))+0.0001)tf.
  
# Detector的损失函数，a=0/1/2，选择损失函数
#loss_Detector=loss_detector(y=label, y_pred=real_lab,a=0)+loss_detector(y=label, y_pred=com_lab,a=0)
loss_Detector=loss_detector(y=label,y_pred=None,mutil1=x1,mutil2=x2,mutil3=x3,y_real=real_lab,y_comb=com_lab,beta=0.6,a=3)+\
              0.6*loss_fuse(x4,x5,x6,label)
#loss_Detector=tf.reduce_mean(tf.abs(label-real_lab))/(tf.reduce_mean(tf.abs(label+real_lab))+0.0001)+\
#   tf.reduce_mean(tf.abs(label-com_lab))/(tf.reduce_mean(tf.abs(com_lab+label))+0.0001)

#测试损失
#下式可以根据m值选择不同的回归损失函数
test_loss_Detector=Regression_loss(y=label,y_pred=real_lab,m=0)




#  各网络参数
t_vars = tf.trainable_variables()

disc_vars = [var for var in t_vars if 'Discriminator' in var.name ]
dete_vars = [var for var in t_vars if 'Detector.' in var.name]  
encoder2_vars = [var for var in t_vars if 'encoder2' in var.name]


#全局迭代次数
global_step = tf.Variable(0, trainable=False)
# 全局epoch数
global_epoch = tf.Variable(0,trainable=False, name='Parameter_name_counter')
# 定义更新epoch操作，计数用
update_epoch = tf.assign(global_epoch, tf.add(global_epoch, tf.constant(1)))


# 学习率 指数衰减衰减
lr = tf.train.exponential_decay(0.0001,global_step,2000,0.9,staircase=True)

#设置优化器
disc_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_D, var_list=disc_vars,global_step=global_step)
encoder2_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_encoder2, var_list=encoder2_vars,global_step=global_step)
Detector_optimizer = tf.train.AdamOptimizer(lr).minimize(loss_Detector, var_list=dete_vars,global_step=global_step)


# 获取训练数据和测试数据,返回文件名list
train_images,test_images = ops.split_image_data(data_dir,train_data_rate,file_postfix='png',shuffle=True)
train_log.info('-----train images:{} ,test images:{} ------'.format(len(train_images),len(test_images)))


train_log.info('------------loading train and test data-------')
# 读取所有的训练集,返回tensor格式
train_data, train_label = reader.load_data(data_dir,train_images,output_size=(w, h),data_enhance= data_enhance)
# 读取所有的测试集,返回tensor格式,测试集不增强
test_data, test_label = reader.load_data(data_dir,test_images,output_size=(w, h),data_enhance=data_enhance)
train_log.info('------------loading success----------')


# 设置训练数据batch形式
train_input_queue = tf.train.slice_input_producer([train_data, train_label], shuffle=True)
train_image_batch, train_label_batch = tf.train.batch(train_input_queue, batch_size=batch_size, num_threads=8,
                                                                capacity=12,allow_smaller_final_batch=True)
# 设置测试数据batch形式
test_input_queue = tf.train.slice_input_producer([test_data, test_label],shuffle=False)
test_image_batch, test_label_batch = tf.train.batch(test_input_queue, batch_size=batch_size,num_threads=8,
                                                                capacity=12,allow_smaller_final_batch=True)



# 定义保存模型操作
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())#这一行必须加，因为slice_input_producer的原因
    coord = tf.train.Coordinator()
    # 启动计算图中所有的队列线程
    threads = tf.train.start_queue_runners(sess,coord)

#     加载检查点，方便二次训练

    train_log.info('----------------------train beginning----------------------------')
    ckpt = tf.train.get_checkpoint_state('./model/')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        current_epoch = int(sess.run(global_epoch))
        train_log.info('Import models successful!  current_epoch: {} current a:{}'.format(current_epoch,a))
#         print('Import models successful!  current_epoch:',current_epoch)
    else:
        sess.run(tf.global_variables_initializer())
        current_epoch = 0
        train_log.info('Init models successful!  current_epoch: {}  current a:{}'.format(current_epoch,a))
#         print('Initialize successful! current_epoch:',current_epoch)

    if current_epoch>=epoch:
         # 主线程计算完成，停止所有采集数据的进程
        coord.request_stop()
        coord.join(threads)
    #     关闭会话
        sess.close()
        train_log.info("已达到循环epoch")
        sys.exit()


     # 主线程，循环epoch
    for i in range(current_epoch,epoch):
  #       计时用
        start = time.time()

# #         循环迭代次数
        for j in range(len(train_images)//batch_size+1):

    #         由于批次类型为张量  这里先使用 run 获取到数据信息后再feed到网络中训练，
            train_feed_image,train_feed_label = sess.run([train_image_batch, train_label_batch])
            noise_z = np.random.uniform(0, 1.0, [batch_size,1,h,w])
            feed_noise_image = np.concatenate((train_feed_image,noise_z),axis=1)

    #         喂入网络的数据
            feeds = {image: train_feed_image, label: train_feed_label,noise_image:feed_noise_image}
    #         训练网络，获取相关信息

            #_loss_encoder2, _= sess.run([ loss_encoder2, encoder2_optimizer],feed_dict=feeds)
            _loss_encoder2, _= sess.run([ loss_encoder2, encoder2_optimizer],feed_dict=feeds)
            _loss_Detector, _,input_image,input_label,output_real_lab,output_com_lab,learningrate = sess.run([ loss_Detector,
                                                                                              Detector_optimizer,
                                                                                               image,label,real_lab,
                                                                                              com_lab,lr],feed_dict=feeds)
            _loss_D, _,current_epoch= sess.run([ loss_D, disc_optimizer,global_step],feed_dict=feeds)



        current_epoch=int(sess.run(update_epoch))

#           打印当前的损失

        end = time.time()
        train_log.info('epoch:{} Detector_loss:{} D_loss:{} encoder2_loss:{} lr:{} runing time:{} s '.format(current_epoch, _loss_Detector,_loss_D,
                                                                                    _loss_encoder2, learningrate,round((end-start),2)))

        

#         10个epoch保存一次相关图片 和模型
        if i%50 == 0 :
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_image' + '.png' , np.transpose(input_image[0], (1,2,0)))
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_label' + '.png' , np.transpose(input_label[0], (1,2,0)))
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_real_lab' + '.png' , np.transpose( output_real_lab[0], (1,2,0)))
            cv2.imwrite('./data_save/record/' + str(current_epoch) + "_" +  '_fake_lab' + '.png' , np.transpose(output_com_lab[0], (1,2,0)))

            saver.save(sess, './model/fcrn.ckpt', global_step=current_epoch)

#         #1000epoch test model

        if i%60 == 0 and i!=0:
            train_log.info("--------test start---------")
            start = time.time()

            for itor in range(len(test_images)//batch_size+1):
                test_feed_data,test_feed_label = sess.run([test_image_batch, test_label_batch])
#                 print('itor ,test_feed_data,test_feed_label shape',itor,test_feed_data.shape,test_feed_label.shape)

            # 喂入网络的数据
                test_feeds = {image: test_feed_data, label: test_feed_label}
                _test_loss_Detector,_test_input_image,_test_input_label,_test_output_real_lab = sess.run([ test_loss_Detector,
                                                                                                 image,label,real_lab],feed_dict=test_feeds)
                for img_idx in range(len(_test_input_image)):
                    cv2.imwrite('./data_save/test/' + str(current_epoch) + '_'+str(itor)+"_" + str(img_idx)+ '_image' + '.png' , np.transpose(_test_input_image[img_idx], (1,2,0)))
                    cv2.imwrite('./data_save/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+  '_label' + '.png' , np.transpose(_test_input_label[img_idx], (1,2,0)))
                    cv2.imwrite('./data_save/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+ '_real_lab' + '.png' , np.transpose(_test_output_real_lab[img_idx], (1,2,0)))
                train_log.info('test save img nums:{}'.format(img_idx+1))

                train_log.info(' test:  epoch:{} itor:{} test_Detector_loss:{} '.format(current_epoch, itor,_test_loss_Detector))
            end = time.time()

            train_log.info("--------test end ,runing time:{} s ---------".format(round((end-start),2)))


#      最后保存一批次图片
    train_log.info("save final batch image!")
#     print("save final batch image!")

    for img_idx in range(len(input_image)):

        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" + str(img_idx)+ '_image' + '.png' , np.transpose(input_image[img_idx], (1,2,0)))
        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+  '_label' + '.png' , np.transpose(input_label[img_idx], (1,2,0)))
        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+ '_real_lab' + '.png' , np.transpose(output_real_lab[img_idx], (1,2,0)))
        cv2.imwrite('./data_save/final/train/' + str(current_epoch) + "_" +str(img_idx)+  '_fake_lab' + '.png' , np.transpose(output_com_lab[img_idx], (1,2,0)))
    train_log.info('final save img nums:{}'.format(img_idx+1))
#     print('final save img nums:',img_idx+1)

    train_log.info("--------final_test start---------")
    start = time.time()

    for itor in range(len(test_images)//batch_size+1):
        test_feed_data,test_feed_label = sess.run([test_image_batch, test_label_batch])
#                 print('itor ,test_feed_data,test_feed_label shape',itor,test_feed_data.shape,test_feed_label.shape)

    # 喂入网络的数据
        test_feeds = {image: test_feed_data, label: test_feed_label}
        _test_loss_Detector,_test_input_image,_test_input_label,_test_output_real_map = sess.run([ test_loss_Detector,
                                                                                         image,label,real_lab],feed_dict=test_feeds)
        for img_idx in range(len(_test_input_image)):
            cv2.imwrite('./data_save/final/test/' + str(current_epoch) + '_'+str(itor)+"_" + str(img_idx)+ '_image' + '.png' , np.transpose(_test_input_image[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+  '_label' + '.png' , np.transpose(_test_input_label[img_idx], (1,2,0)))
            cv2.imwrite('./data_save/final/test/' + str(current_epoch) + '_'+str(itor)+"_" +str(img_idx)+ '_real_lab' + '.png' , np.transpose(_test_output_real_map[img_idx], (1,2,0)))
        train_log.info('test save img nums:{}'.format(img_idx+1))

        train_log.info(' test:  epoch:{} itor:{} test_Detector_loss:{} '.format(current_epoch, itor,_test_loss_Detector))
    end = time.time()

    train_log.info("--------final test end ,runing time:{} s ---------".format(round((end-start),2)))
    train_log.info("-------------------train finish---------------------")
# #     print("Done!")

    saver.save(sess, './model/fcrn.ckpt', global_step=current_epoch)


# 主线程计算完成，停止所有采集数据的进程
    coord.request_stop()
    coord.join(threads)
#     关闭会话
    sess.close()


