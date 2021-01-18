# TEAM: 0045_010_1143
# model.py contains all functions and classes associated with our neural network

#The accuracy in the attached screenshot in our zip file 
# is 90% for test set and 86.8% for train set. 
# Accuracy may vary, we humbly request you to run this file at least thrice!

#importing required libraries
import pandas as pd
import numpy as np

# cleaned csv file already exists in this directory as well as the 'data' directory
dataset = pd.read_csv('Clean_LBW_Dataset.csv')

#dataset = pd.read_csv('Clean_LBW_Dataset.csv')
#np.random.seed(0)
class NN:

    def use_df(self,dataset):
        self.df = dataset
 
    # number of neurons in each layer
    # we have one hidden layer with 16 neurons
    layers = [9, 22, 2]

    # activation functions
    def sig(self,z):
        return (1)/(1 + np.exp(-z))
    def sig_prime(self,z):
        return self.sig(z)*(1-self.sig(z))
    def tanH(self,z):
        return np.tanh(z)
    def tanH_prime(self,z):
        return 1 - ((self.tanH(z))**2)


    # Xavier's initialization of weights and bias
    hidden_weights = np.array(np.random.randn(layers[1], layers[0]))*(np.sqrt(2/(layers[1] + layers[0])))
    hidden_bias = np.zeros(layers[1])
    output_weights = np.array(np.random.randn(layers[2], layers[1]))*(np.sqrt(2/(layers[1] + layers[2])))
    output_bias = np.zeros(layers[2])

    # label encoding to accommodate the attribute's values and make it suitable for the neural network
    def encode(self,inp):
        if inp==0:
            return np.array([1, 0])
        elif inp==1:
            return np.array([0, 1])


    # feed forward functions according to formulas in 'Machine Learning' by Tom M. Mitchell
    # activation function used: tanH
    def feed_forward(self, row):
        x = np.array(self.df.iloc[row, :-1])
        # dot product of the transpose of x (array version of the 'Result' attribute)
        dot_prod1 = np.array(np.dot(self.hidden_weights, x.T)) 
        hidden_act = self.tanH(np.add(dot_prod1, self.hidden_bias)) 
        dot_prod2 = np.dot(self.output_weights, hidden_act)
        #applying the activation function on output_bias (dcelared initially with zeroes)
        output = self.tanH(np.add(dot_prod2, self.output_bias.T))
        return x, output, hidden_act

    # Back Propagation function
    def back_propagation(self, x, row, output, hidden_act, eta):
        #encode function does label encoding on the input row as specified before assigning it to 'expected'
        expected = self.encode(int(self.df.iloc[row,-1]))
        
        #The following errors and derivatives are according to Tom Mitchell's Derivation of the Back Propagation Rule:
        error = expected - output
        d_output = self.tanH_prime(np.array(output))
        d_hidden = self.tanH_prime(np.array(hidden_act))
        delta_output = error*d_output
        error_hidden = np.dot(self.output_weights.T, delta_output)
        delta_hidden = error_hidden*d_hidden

        delta_output = np.reshape(delta_output, (2, 1))
        hidden_act = np.reshape(hidden_act, (self.layers[1], 1))
        dot_prod1 = np.dot(delta_output, hidden_act.T)*(eta)
        #After applying the learning rate, output weights are updated accordingly:
        self.output_weights += dot_prod1

        delta_hidden = np.reshape(delta_hidden, (self.layers[1], 1))
        x = np.reshape(x, (9, 1))
        dot_prod2 = np.dot(delta_hidden, x.T)*(eta)
        #Update hidden weights after applying the learning rate:
        self.hidden_weights += dot_prod2
        # Hidden and output biases combined with the learning rate:
        self.hidden_bias += np.sum(delta_hidden, axis = 0)*(eta)
        self.output_bias += np.sum(delta_output, axis = 0)*(eta)

    #fit function calls feedforward and then performs back propagation
    def fit(self,eta):
        for row in range(len(self.df)):
            x, output, hidden_act = self.feed_forward(row)
            self.back_propagation(x, row, output, hidden_act, eta)

    def predict(self, eta):
        # Confusion matrix with 2 elements in each row because 'Result' is binary
        confusion_matrix = [[0, 0], [0, 0]]
        yhat=[]
        for row in range(len(self.df)):
            x, out, hidden_act = self.feed_forward(row)
            exp = self.encode(int(self.df.iloc[row,-1]))
            yhat.append(out)

            if exp[0] == 1:
                if max(out) == out[0]:
                    confusion_matrix[0][0] += 1
                elif max(out) == out[1]:
                    confusion_matrix[1][0] += 1

            if exp[1] == 1:
                if max(out) == out[0]:
                    confusion_matrix[0][1] += 1
                elif max(out) == out[1]:
                    confusion_matrix[1][1] += 1

        acc = (confusion_matrix[0][0] + confusion_matrix[1][1])/len(self.df)
        if confusion_matrix[1]==[0,0]:
            prec = 0.00
        else:
            prec = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        rec = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])

        return confusion_matrix,yhat,acc,prec,rec


np.random.seed(0)

# loading csv file (dataset)
df = pd.read_csv('Clean_LBW_Dataset.csv')

# splitting dataset into training(80%) and testing(20%) data
df = df.reindex(np.random.permutation(df.index))
#trainDf = df[1:int(0.8*len(df))]
trainDf = df[:int(0.8*len(df))]
testDf = df[int((0.8*len(df))):]

# hyperparameters
eta = 0.013         
epochs = 1500

# creating a new neural network called 'net'
net = NN()

# running the 'fit' function on training data
net.use_df(trainDf)
for e in range(epochs):
    net.fit(eta)

# running the 'predict' function on test data
net.use_df(testDf)
cm,yhat,a,p,r = net.predict(eta)

# printing the required performance metrics for the TEST SET
print('TESTING SET:')
print('Confusion Matrix: ',cm)
print('Accuracy:',a)
if p==0 and r==0:
    f1 = 0.00
else:
    f1 = 2*p*r/(p+r)
print('Precision: %.2f'%p,'Recall: %.2f'%r,'F1 score: %.2f'%f1)

# printing the required performance metrics for the TRAIN SET
print('\nTRAINING SET:')
net.use_df(trainDf)
cm,yhat,a,p,r = net.predict(eta)
print('Confusion Matrix:',cm)
print('Accuracy:',a)
if p==0 and r==0:
    f1 = 0.00
else:
    f1 = (2*p*r)/(p+r)
print('Precision: %.2f'%p,'Recall: %.2f'%r,'F1 score: %.2f'%f1)