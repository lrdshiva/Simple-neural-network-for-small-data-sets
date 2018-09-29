
import numpy as np

class ANN:

    def __init__(self,layers = 2 ,features = 4, h_l_profile = [5 , 5] ,

                 output_class = 2 , max_iter = 10 , alpha = 0.001):


        self. W = { }
        self. B = { }
        self. Z = { }
        self. A = { }
        self.dz = { }
        self.da = { }
        self.dw = { }
        self.max_iter = max_iter
        self.layers = layers
        self.alpha = alpha
        
        if (len(h_l_profile) == 0):
            
            self.W["W0"] = 2 * np.random.random_sample((features,5)) -1
            self.B["B0"] = 2 * np.random.random_sample((1,5)) -1
            for i in range(0,layers-1):

                self.W["W"+str(i+1)] = 2 * np.random.random_sample((5,5)) -1  
                self.B["B"+str(i+1)] = 2 * np.random.random_sample((1,5)) -1

            self.W["W"+str(layers)] = 2 * np.random.random_sample((5,output_class)) -1  
            self.B["B"+str(layers)] = 2 * np.random.random_sample((1,output_class)) -1    
        else:

            self.W["W0"] = 2 * np.random.random_sample((features,h_l_profile[0])) -1
            self.B["B0"] = 2 * np.random.random_sample((1,h_l_profile[0])) -1

            for i in range(0,layers-1):

                self.W["W"+str(i+1)] = 2 * np.random.random_sample((
                                    h_l_profile[i],h_l_profile[i+1])) -1
                self.B["B"+str(i+1)] = 2 * np.random.random_sample((
                                        1         ,h_l_profile[i+1])) -1
                
            self.W["W"+str(layers)] = 2 * np.random.random_sample((h_l_profile[layers-1],output_class)) -1  
            self.B["B"+str(layers)] = 2 * np.random.random_sample((1,output_class)) -1     
        #print(self.W)

    def forward_prop(self,X):

        self.A["A0"] = X

        for l in range(self.layers+1):

            self.Z["Z"+str(l+1)] =  np.dot (self.A["A"+str(l)],self.W["W"+str(l)])+ self.B["B"+str(l)]

            self.A["A"+str(l+1)] =  1/(1 + np.exp(-self.Z["Z"+str(l+1)]))

        return self.A["A"+str(self.layers+1)]



    def back_prop(self,X,t):

        for i in range(self.max_iter):

            self.forward_prop(X)

            error = t.reshape((-1,self.A["A"+str(self.layers+1)].shape[1]))

            error = t - self.A["A"+str(self.layers+1)]

            self.da["da"+str(self.layers+1)] = error

            self.dz["dz"+str(self.layers+1)] = (self.da["da"+str(self.layers+1)] *
                                               (self.A["A"+str(self.layers+1)] *
                                                (1 - self.A["A"+str(self.layers+1)])))

            for l in range(self.layers,-1,-1):

                self.da["da"+str(l)] = self.dz["dz"+str(l+1)].dot(self.W["W"+str(l)].T)
            
                self.dz["dz"+str(l)] = np.multiply( self.da["da"+str(l)] , np.multiply ( self.A["A"+str(l)] , (1 - self.A["A"+str(l)])))

            #weight update

            for l in range(self.layers,-1,-1):

                self.W["W"+str(l)] = np.add( self.W["W"+str(l)] , self.alpha*self.da["da"+str(l)].T.dot(self.dz["dz"+str(l+1)]))
                self.B["B"+str(l)] = np.add( self.B["B"+str(l)] , self.alpha*self.dz["dz"+str(l+1)])

        print("BP done ......")
                          

X = np.array([ [ 0,0,1,1 ] ,
               [ 0,1,1,0 ] ,
               [ 1,0,1,1 ] ,
               [ 1,1,1,0 ] ])
y = np.array([[0,1,0.5],
              [1,0,0],
              [0,1,0],
              [1,0,0.5]])
"""
nn = ANN(layers = 4 ,features = X.shape[1], h_l_profile = [8,5,5,8] ,

                 output_class = y.shape[1] , max_iter = 30 , alpha = 0.01)

print(nn.back_prop(X,y))

print((nn.forward_prop(X)))

"""
X1 = np.array([[248,  89, 149,  80, 175,  42,  18, 132,  55, 197],
       [207,  54,  61,  40,  86,  58, 139, 143, 162, 169],
       [ 52, 195,  51, 165,  47, 219, 250,   0,   5,  61],
       [ 51,  71,  89, 160,  15, 122, 173,  30, 251, 243],
       [ 94,  76, 123,  46,  65, 120, 228,  56,  28, 139],
       [204,  25,  55,  99, 217,  95,  70,  38, 157, 223]])
X1_norm = X1/255
nn1 = ANN(layers = 6 ,
          features = X1.shape[1],
          h_l_profile = [10,7,4,4,7,10] ,
          output_class = X1.shape[1] ,
          max_iter = 20000 ,
          alpha = 1)

print(nn1.back_prop(X1_norm,X1_norm))
print(X1,"\n\n")
print(X1-np.around(255*(nn1.forward_prop(X1_norm))))

