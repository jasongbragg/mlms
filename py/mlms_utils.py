def freq2gt(af, f=0):
   import numpy as np
   ind_gt=[]
   for i in range(len(af)):
      exp_h=2*af[i]*(1-af[i]) 
      prb_h=exp_h*(1-f)
      prb_r=(1-prb_h)*(af[i]**2 / (af[i]**2 + (1-af[i])**2) ) 
      prb_a=1-prb_h-prb_r
      ind_gt.append(np.random.choice([1,2,0],p=[prb_h,prb_r,prb_a]))
   return ind_gt         

def gterrors(gt, het2hom=0.01, hom2het=0.001, hom2hom=0.001):
   import numpy as np
   new_gt=[]
   for i in range(len(gt)):
      if gt[i] == 1:
         ngt=np.random.choice([1,0,2],p=[1-het2hom, het2hom*0.5, het2hom*0.5])
      if gt[i] == 0:
         ngt=np.random.choice([0,1,2],p=[1-hom2het-hom2hom, hom2het, hom2hom])
      if gt[i] == 2:
         ngt=np.random.choice([2,1,0],p=[1-hom2het-hom2hom, hom2het, hom2hom])
      new_gt.append(ngt)
   return new_gt         

def self2gt(gt):
   import numpy as np
   new_gt=[]
   for i in range(len(gt)):
      if gt[i] == 1:
         ngt=np.random.choice([1,0,2],p=[0.5, 0.25, 0.25])
      if gt[i] == 0:
         ngt=0
      if gt[i] == 2:
         ngt=2
      new_gt.append(ngt)
   return new_gt         

def parents2gt(g1, g2):
   import numpy as np
   new_gt=[]
   for i in range(len(g1)):
      if g1[i] == 0 and g2[i] == 0 :
         ngt=0
      if g1[i] == 2 and g2[i] == 2 :
         ngt=2
      if g1[i] == 1 and g2[i] == 1 :
         ngt=np.random.choice([1,0,2],p=[0.5, 0.25, 0.25])
      if g1[i] == 0 and g2[i] == 1 :
         ngt=np.random.choice([1,0],p=[0.5, 0.5])
      if g1[i] == 1 and g2[i] == 0 :
         ngt=np.random.choice([1,0],p=[0.5, 0.5])
      if g1[i] == 2 and g2[i] == 1 :
         ngt=np.random.choice([2,1],p=[0.5, 0.5])
      if g1[i] == 1 and g2[i] == 2 :
         ngt=np.random.choice([2,1],p=[0.5, 0.5])
      if g1[i] == 2 and g2[i] == 0 :
         ngt=1
      if g1[i] == 0 and g2[i] == 2 :
         ngt=1
      new_gt.append(ngt)
   return new_gt         

def kin2gt(gt, af, k):
   import numpy as np
   new_gt=[]
   for i in range(len(gt)):
      # get a1
      a1 = -9
      if gt[i] == 1:
         a1=np.random.choice([1,0],p=[0.5, 0.5])
      if gt[i] == 0:
         a1=0
      if gt[i] == 2:
         a1=1
      # get a2
      a2 = -9
      a2_ibd=np.random.choice([1,0],p=[k, 1-k])
      if a2_ibd == 1:
         if gt[i] == 1:
            a2=np.random.choice([1,0],p=[0.5, 0.5])
         if gt[i] == 0:
            a2=0
         if gt[i] == 2:
            a2=1
      if a2_ibd == 0:
            a2=np.random.choice([1,0],p=[af[i], 1-af[i]])
      ngt=a1+a2
      new_gt.append(ngt)
   return new_gt         


def generate_outcrossed(af, pop_freq, samp_array, n):
   import numpy as np
   for i in range(n):
      moth_rl  = freq2gt(af)
      moth_ind = gterrors(moth_rl)
      fath_rl = freq2gt(af)
      prog_ind = gterrors(parents2gt(moth_rl, fath_rl))
      moth_rsc = [i * 50 for i in moth_ind]
      prog_rsc = [i * 50 for i in prog_ind]
      samp_mat = np.array([moth_rsc, prog_rsc, pop_freq], dtype=np.uint8)
      samp_array=np.append(samp_array,[samp_mat],axis=0)
   return samp_array


def generate_apomictic(af, pop_freq, samp_array, n):
   import numpy as np
   for i in range(n):
      moth_rl = freq2gt(af)
      moth_ind = gterrors(moth_rl)
      prog_ind = gterrors(moth_rl)
      moth_rsc = [i * 50 for i in moth_ind]
      prog_rsc = [i * 50 for i in prog_ind] 
      samp_mat = np.array([moth_rsc, prog_rsc, pop_freq], dtype=np.uint8)
      samp_array=np.append(samp_array,[samp_mat],axis=0)
   return samp_array

def generate_selfed(af, pop_freq, samp_array, n):
   import numpy as np
   for i in range(n):
      moth_rl = freq2gt(af)
      moth_ind = gterrors(moth_rl)
      prog_ind = gterrors(self2gt(moth_rl))
      moth_rsc = [i * 50 for i in moth_ind]
      prog_rsc = [i * 50 for i in prog_ind]
      samp_mat = np.array([moth_rsc, prog_rsc, pop_freq], dtype=np.uint8)
      samp_array=np.append(samp_array,[samp_mat],axis=0)
   return samp_array

def generate_bipar_ibd(af, pop_freq, samp_array, n, k):
   import numpy as np
   for i in range(n):
      moth_rl = freq2gt(af)
      moth_ind = gterrors(moth_rl)
      prog_ind = gterrors(kin2gt(moth_rl, af, k))
      moth_rsc = [i * 50 for i in moth_ind]
      prog_rsc = [i * 50 for i in prog_ind]
      samp_mat = np.array([moth_rsc, prog_rsc, pop_freq], dtype=np.uint8)
      samp_array=np.append(samp_array,[samp_mat],axis=0)
   return samp_array
