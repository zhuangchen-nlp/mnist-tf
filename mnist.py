import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE  = 100
INPUT_NODE  = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
TRAIN_STEP  = 10000
LEARNING_RATE = 0.01
L2NORM_RATE   = 0.001


def train(mnist):
    # define input placeholder
    input_x = tf.placeholder( tf.float32, [None,INPUT_NODE],  name = "input_x")
    input_y = tf.placeholder( tf.float32, [None,OUTPUT_NODE], name = "input_y")

    #define weights and biases
    w1 = tf.Variable( tf.truncated_normal(shape = [INPUT_NODE,LAYER1_NODE], stddev = 0.1) )
    b1 = tf.Variable( tf.constant(0.1, shape= [LAYER1_NODE]) )

    w2 = tf.Variable( tf.truncated_normal(shape = [LAYER1_NODE,OUTPUT_NODE], stddev = 0.1) )
    b2 = tf.Variable( tf.constant(0.1, shape= [OUTPUT_NODE]) )

    layer1 = tf.nn.relu( tf.nn.xw_plus_b(input_x, w1, b1) )
    y_hat  = tf.nn.xw_plus_b(layer1, w2 ,b2)
    print("1 step ok!")

    #define loss
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits( logits=y_hat, labels= input_y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularization = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) +tf.nn.l2_loss(b1) +tf.nn.l2_loss(b2)
    loss = cross_entropy_mean + L2NORM_RATE*regularization
    print("2 step ok!")

    #define accuracy
    correct_predictions = tf.equal(tf.argmax(y_hat,1),tf.argmax(input_y,1))
    accuracy  = tf.reduce_mean( tf.cast( correct_predictions,tf.float32) )
    print("3 step ok!")

    #train operation
    global_step = tf.Variable( 0, trainable= False)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,global_step=global_step)
    print("4 step ok!")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("5 step ok!")

        for i in range(TRAIN_STEP):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            #print("ys shape:{}".format(np.shape(ys)))
            feed_dict = {
                input_x : xs,
                input_y : ys
            }

            _, step, train_loss, train_acc = sess.run([train_op, global_step, loss, accuracy], feed_dict=feed_dict)
            if (i % 100 == 0):
                print("After %d steps, in train data, loss is %g, accuracy is %g." % (step, train_loss, train_acc))

        test_feed = {input_x: mnist.test.images, input_y: mnist.test.labels}
        test_acc = sess.run(accuracy,feed_dict = test_feed)
        print("After %d steps, in test data, accuracy is %g." % (TRAIN_STEP, test_acc))


if __name__ == "__main__":
    mnist = input_data.read_data_sets("data", one_hot= True)
    print("0 step ok!")
    train(mnist)
