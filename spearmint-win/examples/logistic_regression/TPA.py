import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import SGD
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

def logitreg(space):
    rate = space['rate']
    l2_reg = space['l2_reg']
    batchsize = space['batchsize']
    num_epochs = space['num_epochs']
    
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
    acclist=[-1,-1,-1]
    for epoch in range(num_epochs):
        logitreg.fit(x_train, y_train, batch_size=batchsize, epochs=1, verbose=0)
        pred=logitreg.predict(x_test)
        y_pred=np.argmax(pred,1)
        acc=accuracy_score(y_test,y_pred)
        if(abs(acc-acclist[-1])<1e-3):
            break
        else:
            acclist.append(acc)
    return(1-acc)


space = {
    'rate': hp.uniform('rate', 0, 1),
    'l2_reg': hp.uniform('l2_reg', 0, 1),
    'batchsize' : hp.randint('batchsize', 20, 2000),
    'num_epochs' : hp.randint('num_epochs', 5, 2000)
}


result=np.zeros((10,100))
for i in range(10):
    tpe_algo = tpe.suggest
    tpe_trials = Trials()
    tpe_best = fmin(fn=logitreg, space=space, 
                    algo=tpe_algo, trials=tpe_trials, 
                    max_evals=100)
    funcvalue=np.zeros(100)
    for j in range(100):
        funcvalue[j]=tpe_trials.results[j]['loss']
    minvalue=np.zeros(100)
    minvalue[0]=funcvalue[0]
    for j in range(1,100):
        minvalue[j]=min(minvalue[j-1],funcvalue[j])
    result[i,:]=minvalue
    #np.savetxt('tpe'+str(i)+'.csv', minvalue, delimiter=',')
    
mu_tpe=np.mean(result,0)
std_tpe=np.std(result,0)
    
    