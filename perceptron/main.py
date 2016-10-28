import numpy as np
import random
%matplotlib inline
import matplotlib.pyplot as plt
from mlxtend.evaluate import plot_decision_regions
import pandas as pd

class Perceptron(object):
    
    def __init__(self, n, lr = 0.01, epochs = 100):
        self.lr = lr
        self.epochs = epochs
        self.n = n
    
    def train(self):
        X, y = self.prepData()
        #self.w = np.random.rand(1, len(X[1])+1)
        #self.w  = [(x-0.5) for x in self.w]
        #print len(self.w)
        #self.w = self.w[0]
        self.w = np.random.uniform(-0.5, 0.5, len(X[1])+1)
        self.errors = []
        for _ in range(0, self.epochs):
            error = 0
            for x, target in zip(X, y):
                #print x
                coefUpd = self.lr *(target - self.pred(x))
                self.w[1:] += x*coefUpd
                self.w[0] +=coefUpd
                
                error +=int(coefUpd != 0.0) 
            self.errors.append(error)
            if(_%25 == 0):
                print ("Epoch: " + str(_) +", Error: "+str(error))
            
        return self
    
    def test(self):
        mat = [[0,0],[0,0]]
        dt = pd.read_csv("mnist_test.csv")
        y = dt.iloc[:,0].values
        y = np.where(y!=self.n, 0, 1)
        X = dt.iloc[:,1:].values
        
        for x, i in zip(X,y):
            if int(self.pred(x)) == i and i == 1: 
                mat[0][0]+=1
            elif int(self.pred(x)) == i and i == 0:
                mat[1][1]+=1
            elif int(self.pred(x)) != i and i == 1:
                mat[0][1]+=1
            elif int(self.pred(x)) != i and i == 0:
                mat[1][0]+=1
        
        return mat
        
                
    def step(self, dp):
        return np.where(dp >= 0, 1, 0)
    
    def dp(self, X):
        return np.dot(X,self.w[1:])+self.w[0]
    
    def pred(self, xi):
        return self.step(self.dp(xi))
    def predN(self, xi):
        return self.dp(xi)
    
    def prepData(self):
        tr = pd.read_csv("mnist_train.csv")
        #for some reason pandas give me the number 5 as the index to first
        #column my guess is 5 appears more in the data
        y = tr.iloc[:,0].values
        labels = y
        X = tr.iloc[:,1:].values
        y = np.where(y!=self.n, 0, 1)
        ls = [[]]
        ly = []
        print "Preparing it"
        for i,l in zip(y,X):
            if(i==1):
                ls.append(list(l))
                ly.append(int(i))
        ls.remove([])
      
        print "Extending it"
        for i in range(0,8):
            X = np.append(X, ls,axis = 0)
            y = np.append(y, ly, axis = 0)
            c = list(zip(X, y))
            random.shuffle(c)
            X, y = zip(*c)
        
        return X, y
pctr = Perceptron(0, 0.001,10)
pctr.train()
#print('Weights: %s' % pctr.w)
#plot_decision_regions(X, y, clf=pctr)
#plt.title('Perceptron')
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.show()
#print pctr.w
plt.plot(range(1, len(pctr.errors)+1), pctr.errors, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Missclassifications')
plt.show()
import cv2
from IPython.display import Image, HTML, display
from glob import glob

ls = glob('imagens/*')
#print ls
imagesList=''.join( ["<img style='width: 120px; margin: 0px; float: left; border: 1px solid black;' src='%s' />" % str(s) 
                     for s in sorted(ls[:2]) ])
cont = 0
for i in ls:
    img  = cv2.imread(i,0)
    
    if (pctr.pred(img.flatten()) == 1):
        cont+=1
        imagesList+=("<img style='width: 120px; margin: 0px; float: left; border: 1px solid black;' src='%s' />" % str(i))
        
#uncomment to display all N's values founded 

#display(HTML(imagesList))

print pctr.test()
mnist = []
for i in range(0,10):
    mnist.append(Perceptron(i, 0.001, 200))
    mnist[i].train()
    plt.plot(range(1, len(mnist[i].errors)+1), mnist[i].errors, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Missclassifications')
    plt.show()
tot = 0
ct = 0
test = pd.read_csv("mnist_test.csv")
for i, line in zip(ls, range(0,9999)):
    mai = 0
    ind = -1
    img  = cv2.imread(i,0)
    for index,clf in zip(range(0,10),mnist):
        p = clf.predN(img.flatten())
        if p > mai:
            mai = p
            ind = index
    if(ind == test.iloc[line,0]):
        #print test.iloc[line, 0], ind
        ct+=1
    
    tot+=1
    
print float((ct/float(tot)*100))
print ct, tot
                    
for index,clf in zip(range(0,10),mnist):
    print "\n###########\nPara o "+str(index)
    mat = clf.test()
    print mat
    print "Precision: "+str(float(mat[0][0]/float(mat[0][0]+mat[0][1]))*100)
    print "Accuracy: "+str(float(mat[0][0]+mat[1][1])/float(mat[0][0]+mat[0][1]+mat[1][1]+mat[1][0])*100)
    
print 986/float(10000)
import pickle

# write python dict to a file
output = open('myfile.pkl', 'wb')
for i in mnist:
    
    pickle.dump(i.w, output)
    output.close()

# read python dict back from the file
#pkl_file = open('myfile.pkl', 'rb')
#mydict2 = pickle.load(pkl_file)
#pkl_file.close()

#print mydict
#print mydict2



