import numpy as np


X = np.array([ [ 0,0,1,1 ] ,
               [ 0,1,1,0 ] ,
               [ 1,0,1,1 ] ,
               [ 1,1,1,0 ] ])
y = np.array([[0,1],
              [1,0],
              [0,1],
              [1,0]])

W0 = 2 * np.random.random_sample((4,10)) -1
B0 = 2 * np.random.random_sample((1,10)) -1
W1 = 2 * np.random.random_sample((10,5)) -1
B1 = 2 * np.random.random_sample((1,5)) -1
W2 = 2 * np.random.random_sample((5,10)) -1
B2 = 2 * np.random.random_sample((1,10)) -1
W3 = 2 * np.random.random_sample((10,2)) -1
B3 = 2 * np.random.random_sample((1,2)) -1
print(W0)

def sig(z):
  return 1/(1 + np.exp(-z))
def relu(z):
    
    return (z>=0.0)*z
print(y," ",y.reshape(4,2))
for j in range(60000):

  #forward propagation
  z1 = (np.dot (X,W0) + B0)
  a1 = 1/(1 + np.exp(-z1))
  z2 = np.dot (a1,W1) + B1
  a2 = 1/(1 + np.exp(-z2))
  z3 = np.dot (a2,W2) + B2
  a3 = 1/(1 + np.exp(-z3))
  z4 = np.dot (a3,W3) + B3
  #a4 = relu (z4)
  a4 = 1/(1 + np.exp(-z4))
  #print("a2",a2.shape)
    
  #back propagation
  da4 = y.reshape(-1,2) - a4 
  dz4 = (da4 * (a4 * (1 - a4)))
  
  da3 = dz4.dot(W3.T)
  dz3 = np.multiply( da3 , np.multiply ( a3 , 1 - a3 ))
  
  da2 = dz3.dot(W2.T) 
  dz2 = np.multiply( da2 , np.multiply ( a2 , 1 - a2 ))
  
  da1 = dz2.dot(W1.T)
  dz1 = np.multiply( da1 , np.multiply ( a1 , 1 - a1 ))
  
  W3 = np.add( W3 , a3.T.dot(dz4))
  B3 = np.add( B3 , dz4)
  W2 = np.add( W2 , a2.T.dot(dz3))
  B2 = np.add( B2 , dz3)
  W1 = np.add( W1 , a1.T.dot(dz2))
  B1 = np.add( B1 , dz2)
  W0 = np.add( W0 , X.T.dot(dz1))
  B0 = np.add( B0 , dz1)
  if (j>59990):
    print(np.sum(np.square(da4)))
print((np.around(a4)))

