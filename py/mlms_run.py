import numpy as np
import random
from mlms_utils import freq2gt, parents2gt, gterrors, self2gt, kin2gt, generate_outcrossed, generate_apomictic, generate_selfed, generate_bipar_ibd
from keras.utils import to_categorical

t_each=1000
t_out=t_each
t_apo=t_each
t_slf=t_each
t_bpi=t_each

v_each=200
v_out=v_each
v_apo=v_each
v_slf=v_each
v_bpi=v_each

nloc=400
ncat=4
pop_freq  = [random.randint(0,100) for i in range(nloc)]
af=[x / float(100) for x in pop_freq]


############################################################
### Make simulated data sets

####################
### training set

# initialie
Xt_samp_array= np.empty((0,3,nloc), dtype=np.uint8)

# add different categories
Xt_samp_array = generate_outcrossed(af, pop_freq, Xt_samp_array, t_out)
Xt_samp_array = generate_apomictic(af, pop_freq, Xt_samp_array, t_apo)
Xt_samp_array = generate_selfed(af, pop_freq, Xt_samp_array, t_slf)
Xt_samp_array = generate_bipar_ibd(af, pop_freq, Xt_samp_array, t_bpi, 0.5)

####################
### validation set

# initialize
Xv_samp_array= np.empty((0,3,nloc), dtype=np.uint8)

# add different categories
Xv_samp_array = generate_outcrossed(af, pop_freq, Xv_samp_array, v_out)
Xv_samp_array = generate_apomictic(af, pop_freq, Xv_samp_array, v_apo)
Xv_samp_array = generate_selfed(af, pop_freq, Xv_samp_array, v_slf)
Xv_samp_array = generate_bipar_ibd(af, pop_freq, Xv_samp_array, v_bpi, 0.5)

Xt = Xt_samp_array.reshape(Xt_samp_array.shape[0],3,nloc,1)
Xv = Xv_samp_array.reshape(Xv_samp_array.shape[0],3,nloc,1)


############################################################
### Make labels for datasets


t_cat_vec=[0]*t_out + [1]*t_apo + [2]*t_slf + [3]*t_bpi 
t_cat_array= np.array(t_cat_vec, dtype=np.uint8)

v_cat_vec=[0]*v_out + [1]*v_apo + [2]*v_slf + [3]*v_bpi 
v_cat_array= np.array(v_cat_vec, dtype=np.uint8)

Yt = to_categorical(t_cat_array)
Yv = to_categorical(v_cat_array)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

model = Sequential()

model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(3,nloc,1)))
model.add(Flatten())
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(Xt, Yt, validation_data=(Xv, Yv), epochs=3)

Yp = model.predict_classes(Xv)

import tensorflow as tf
con_mat = tf.math.confusion_matrix(labels=v_cat_array, predictions=Yp).numpy()

#>>> con_mat
#array([[182,   0,   0,  18],
#       [  0, 200,   0,   0],
#       [  0,   1, 198,   1],
#       [  4,   0,   8, 188]], dtype=int32)
### in the above, row: real category
#                 col: pred cstegory



