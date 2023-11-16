import numpy as np
import pandas as pd 
import time
import logging
import params
import dataloader
import gc
from utils2.utils import timeSince,save_logs,simple_evaluate
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from utils2.perturb import timeDelayAttack
import sys

# Preprocess

val = [4,5]
test = [6,7]
ho = list(set(val + test))
train_dir = [params.root_dir+'/data/sim'+str(i+1)+'.csv' for i in range(25) if i+1 not in ho]
val_dir = [params.root_dir+'/data/sim'+str(i)+'.csv' for i in val]
test_dir = [params.root_dir+'/data/sim'+str(i)+'.csv' for i in test]

x_train,y_train = dataloader.dataloader(train_dir)
x_val,y_val = dataloader.dataloader(val_dir)
x_test,y_test = dataloader.dataloader(test_dir, attack_fcn=timeDelayAttack)
input_shape = (x_train.shape[1],x_train.shape[2])

import keras.backend as K
import tensorflow as tf 
sess = K.get_session()


print('x_train shape: '+str(x_train.shape)+';y_train shape: '+str(y_train.shape)) #x_train shape: (100, 8, 80);y_train shape: (100, 160)
print('x_val shape: '+str(x_val.shape)+';y_val shape: '+str(y_val.shape))        #x_val shape: (100, 8, 80);y_val shape: (100, 160)
print('x_test shape: '+str(x_test.shape)+';y_test shape: '+str(y_test.shape))    #x_test shape: (100, 8, 80);y_test shape: (100, 160)



#y_train = np.transpose(y_train)
# Load the models
if params.model_type == 'fcn':
    from regressor import fcn
    regressor = fcn.Regressor_FCN(params.output_directory, input_shape,verbose=True)
elif params.model_type == 'resnet':
    from regressor import resnet
    regressor = resnet.Regressor_RESNET(params.output_directory, input_shape)
elif params.model_type == 'cnn':
    from regressor import cnn
    regressor = cnn.Regressor_CNN(params.output_directory, input_shape)
else:
    logging.error("Invalid mode_type; It could only be one of 'fcn','resnet','cnn'")


# Initialize adversarial example with input image
x_adv = x_test
# Added noise
x_noise = np.zeros_like(x_test[0])
# Set variables
epochs = 100
epsilon = 0.01
target_class = 1 # cucumber
prev_probs = []

for i in range(epochs): 
    # One hot encode the target class
    target = K.variable(y_test)
    
    # Get the loss and gradient of the loss wrt the inputs
    loss = -1*K.square(target-regressor.model.output)
    grads = K.gradients(loss, regressor.model.input)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    x_noise = x_noise + delta

    # Perturb the image
    x_adv = x_adv + epsilon*delta

    # Get the new image and predictions
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    x_adv = sess.run(x_adv, feed_dict={regressor.model.input:x_test})
    preds = regressor.model.predict(x_adv)

    # Store the probability of the target class
    prev_probs.append(preds[0][target_class])

    if i%20==0:
        print(i, preds[0][target_class])


#Training
if params.TRAIN_MODEL:
    hist = regressor.fit(x_train,y_train,x_val, y_val)
    save_logs(hist=hist)
else:
    save_logs()
y_pred = regressor.model.predict(x_adv)
print('Mean squared error in test set: '+str(mean_squared_error(y_test,y_pred)**0.5))

a1,a2 = simple_evaluate(y_test,y_pred)

from sklearn.metrics import mean_squared_error
loss = mean_squared_error(y_test,y_pred)