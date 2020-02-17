import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score
import time


def logitreg(rate,l2_reg,batchsize,num_epochs):
    input_shape=(784,)
    output_dim=10
    
    inputs = tf.keras.Input(shape=input_shape)
    y_hat = tf.keras.layers.Dense(output_dim, kernel_regularizer=regularizers.l2(l2_reg),
                                  activation='softmax')(inputs)
    logitreg = tf.keras.models.Model(inputs, y_hat)
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train=x_train.reshape((-1,784))    
    x_test=x_test.reshape((-1,784))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255


    opt=SGD(lr=np.exp(rate))
    logitreg.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    acclist=[0]
    for epoch in range(num_epochs):
        logitreg.fit(x_train, y_train, batch_size=batchsize, epochs=1, verbose=0)
        pred=logitreg.predict(x_test)
        y_pred=np.argmax(pred,1)
        acc=accuracy_score(y_test,y_pred)
        if(epoch>=4 and abs(acc-acclist[-1])<1e-3):
            break
        else:
            acclist.append(acc)
    return(1-acc)


def main(job_id, params):
    rate = params['RATE'][0]
    l2_reg = params['REG'][0]
    batchsize = params['BATCH'][0]
    num_epochs = params['EPOCH'][0]
    res = logitreg(rate,l2_reg,batchsize,num_epochs)
    print "Logistic regression on mnist:", str(job_id)
    print "\tf(%.2f, %0.2f,%.2f, %0.2f) = %f" % (rate, l2_reg, batchsize, num_epochs, res)
    print time.gmtime()[3:6]
    return res
