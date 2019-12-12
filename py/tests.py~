import numpy as np
import random
from mlms_utils import freq2gt, parents2gt, gterrors, self2gt, kin2gt, generate_outcrossed, generate_apomictic, generate_selfed, generate_bipar_ibd
from keras.utils import to_categorical

n=200

nloc=1000
pop_freq  = [random.randint(0,100) for i in range(nloc)]
af=[x / float(100) for x in pop_freq]



# initialize
samp_array= np.empty((0,3,nloc), dtype=np.uint8)

# add different categories
samp_array_out = generate_outcrossed(af, pop_freq, samp_array, n)
samp_array_apo = generate_apomictic(af, pop_freq, samp_array, n)
samp_array_slf = generate_selfed(af, pop_freq, samp_array, n)
samp_array_b00 = generate_bipar_ibd(af, pop_freq, samp_array, n, 0)
samp_array_b05 = generate_bipar_ibd(af, pop_freq, samp_array, n, 0.5)
samp_array_b10 = generate_bipar_ibd(af, pop_freq, samp_array, n, 1.0)



sum(Xv_samp_array[798][0]==Xv_samp_array[798][1])

out=np.empty((n,6), dtype=np.uint8)

for i in range(n):
   out[i,0] = sum(samp_array_out[i][0]==samp_array_out[i][1])
   out[i,1] = sum(samp_array_apo[i][0]==samp_array_apo[i][1])
   out[i,2] = sum(samp_array_slf[i][0]==samp_array_slf[i][1])
   out[i,3] = sum(samp_array_b00[i][0]==samp_array_b00[i][1])
   out[i,4] = sum(samp_array_b05[i][0]==samp_array_b05[i][1])
   out[i,5] = sum(samp_array_b10[i][0]==samp_array_b10[i][1])

np.mean(out,axis=0)
# array([157.78 , 222.365,  65.76 , 157.16 , 223.105,  63.035])

