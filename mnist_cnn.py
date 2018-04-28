import tensorflow as tf
import numpy as np
from  tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE    = 100
INPUT_SIZE    = 28
FILTER1_SIZE  = 5
FILTER1_NUM   = 32
POOLING2_SIZE = 14
POOLING2_NUM  = 32
FILTER3_SIZE  = 5
FILTER3_NUM   = 64
POOLING4_SIZE = 7
POOLING4_NUM  = 64
FLAT_SIZE     = 3136
FC5_SIZE      = 523
OUTPUT_SIZE   = 10
LEARNING_RATE = 0.001
L2NORM_RATE   = 0.0001
TRAIN_STEP    = 3000

#定义W初始化函数
def get_weights(shape):
    form = tf.truncated_normal(shape,stddev= 0.1)
    return tf.Variable(form)

#定义b初始化函数
def get_biases(shape):
    form = tf.constant(0.1,shape = shape)
    return tf.Variable(form)

def train(mnist):

    #定义输入输出占位
    input_x = tf.placeholder(tf.float32, [None,INPUT_SIZE,INPUT_SIZE,1 ], name= "input_x")
    input_y = tf.placeholder(tf.float32, [None,OUTPUT_SIZE], name= "input_y")
    dropout_keep_prob = tf.placeholder(tf.float32,name = "dropout_keep_prob")
    l2_loss = tf.constant(0.0)
    print("1 step ok!")

    #第一层：卷积层conv1
    '''
    input  : [-1,28,28,1]
    filter : [5,5,32]
    output : [-1,28,28,32]      
    '''
    with tf.name_scope("conv1"):
        w = get_weights([FILTER1_SIZE,FILTER1_SIZE,1,FILTER1_NUM])
        b = get_biases([FILTER1_NUM])
        conv1_op = tf.nn.conv2d(
            input  = input_x,
            filter = w,
            strides = [1,1,1,1],
            padding = "SAME",
            name = 'conv1_op')

        conv1 = tf.nn.relu( tf.nn.bias_add(conv1_op,b) ,name = "relu")
    print("2 step ok!")

    #第二层：持化层pooling2
    '''
    input  : [-1,28,28,32]
    output : [-1,14,14,32]      
    '''
    with tf.name_scope("pooling2"):
        pooling2 = tf.nn.max_pool(
            value = conv1,
            ksize = [1,2,2,1],
            strides = [1,2,2,1],
            padding = "SAME",
            name = "pooling1")
    print("3 step ok!")

    #第三层：卷积层conv3
    '''
    input  : [-1,14,14,32]
    filter : [5,5,64]
    output : [-1,14,14,64]      
    '''
    with tf.name_scope("conv3"):
        w = get_weights([FILTER3_SIZE,FILTER3_SIZE,FILTER1_NUM,FILTER3_NUM])
        b = get_biases([FILTER3_NUM])
        conv3_op = tf.nn.conv2d(
            input = pooling2,
            filter = w ,
            strides = [1,1,1,1],
            padding = "SAME",
            name = "conv3_op")

        conv3 = tf.nn.relu( tf.nn.bias_add(conv3_op,b) ,name = "relu" )
    print("4 step ok!")

    #第四层：池化层pooling4
    '''
    input  : [-1,14,14,64]
    output : [-1,7,7,64]      
    '''
    with tf.name_scope("pooling4"):
        pooling4 = tf.nn.max_pool(
            value = conv3,
            ksize = [1,2,2,1],
            strides = [1,2,2,1],
            padding = "SAME",
            name = "pooling4")

    #池化结果展开
    '''
    input  : [-1,7,7,64]
    output : [-1,3136]      
    '''
    pooling4_flat = tf.reshape(pooling4,[-1,FLAT_SIZE])
    print("5 step ok!")

    #第五层：全连接层fc5
    '''
    input  : [-1,3136]
    output : [-1,512]
    '''
    with tf.name_scope("fc5"):
        w = get_weights([FLAT_SIZE,FC5_SIZE])
        b = get_biases([FC5_SIZE])
        fc5 = tf.nn.relu(tf.nn.xw_plus_b(pooling4_flat,w,b,name=  "fc5"), name = "relu")
        fc5_drop = tf.nn.dropout( fc5,dropout_keep_prob)
        l2_loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    print("6 step ok!")

    #第六层：全连接层（输出）
    '''
    input  : [-1,512]
    output : [-1,10]      
    '''
    with tf.name_scope("fc6"):
        w = get_weights([FC5_SIZE,OUTPUT_SIZE])
        b = get_biases([OUTPUT_SIZE])
        y_hat = tf.nn.xw_plus_b(fc5_drop,w,b,name = "y_hat")
        l2_loss += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    print("7 step ok!")

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = input_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + L2NORM_RATE * l2_loss
    print("8 step ok!")

    correct_predictions = tf.equal(tf.argmax(y_hat,1),tf.argmax(input_y,1))
    accuracy = tf.reduce_mean( tf.cast(correct_predictions,tf.float32) )

    global_step = tf.Variable(0,trainable=False)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,global_step = global_step)
    print("9 step ok!")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEP):
            xs_pre, ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(xs_pre,[-1,INPUT_SIZE,INPUT_SIZE,1])
            feed_dict = {
                input_x: xs,
                input_y: ys,
                dropout_keep_prob : 0.5
            }

            _, step, train_loss, train_acc = sess.run([train_op, global_step, loss, accuracy],feed_dict = feed_dict)

            if i%100 == 0:
                print("step:{} ,train loss:{:g}, train_acc:{:g}".format(step,train_loss,train_acc))

        test_x = np.reshape(mnist.test.images[:1000],[-1,INPUT_SIZE,INPUT_SIZE,1])
        test_y = mnist.test.labels[:1000]
        feed_test = {
            input_x : test_x,
            input_y : test_y,
            dropout_keep_prob : 1.0
        }
        test_loss, test_acc = sess.run([loss,accuracy],feed_dict = feed_test)
        print("After {} training steps, in test dataset, loss is {:g}, acc is {:g}".format(TRAIN_STEP,test_loss,test_acc))

if __name__ == "__main__":
    mnist = input_data.read_data_sets("data",one_hot= True)
    print("0 step ok!")
    train(mnist)
