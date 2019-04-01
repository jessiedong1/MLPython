"""
This program is answering the question 1 of the Midterm
The best subset is {'sepal width', 'petal length'} which has accuracy on training dataset = 0.914; accuracy on testing dataset = 0.956
The details are as follows
"""
import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(100)
    dataset = load_dataset()
    print(dataset.columns)
# Get the subsets of the features (0,1,2,3)
# There are 6 combinations of features
# (0,1) acc on training dataset = 0.771; acc on testing dataset = 0.689
    #set = feature_sub(0, 1, dataset)
# (0,2) acc on training dataset = 0.933; acc on testing dataset = 0.911
    #set = feature_sub(0, 2, dataset)
# (0,3) acc on training dataset = 0.924; acc on testing dataset = 0.867
    #set = feature_sub(0, 3, dataset)
# (1,2) acc on training dataset = 0.914; acc on testing dataset = 0.956
    set = feature_sub(1, 2, dataset)
# (1,3) acc on training dataset = 0.943; acc on testing dataset = 0.911
    #set = feature_sub(1, 3, dataset)
# (2,3) acc on training dataset = 0.895; acc on testing dataset = 0.711
    #set = feature_sub(2, 3, dataset)


    train_set, test_set = split_train_test(set)
    numHiddens = 10
    eth = 0.01
    numOutputs = 3
    num_runs = 190


    ACC_Train, ACC_Test = MLP(numHiddens, numOutputs, train_set,eth,num_runs,test_set)
    print("Accuracy on Trainset: {}".format(ACC_Train))
    print("Accuracy on Tesetset: {}".format(ACC_Test))

    plotResult(ACC_Train, ACC_Test,num_runs,eth)
    #test_MLP(ih_weights, ho_weights, test_set)

def load_dataset():
    dataset = load_iris()
    data = dataset['data']
    label = dataset['target']
    atts = ['sepal length', 'sepal width','petal length', 'petal width']
    dataset = pd.DataFrame(data, columns=atts)
    dataset['Class'] = label
    return dataset

#Get the subsets of the features (0,1,2,3)
def feature_sub(fea1, fea2, dataset):
    #ROWS = dataset.shape[0]
    #bias = [1]*ROWS
    testset = dataset.iloc[:, [fea1, fea2,4]]
    #testset.insert(loc = 0, column = 'bias', value = bias)
    #testset['bias'] = bias
    return testset

def split_train_test(dataset):
    X_1_train = dataset.iloc[0:35,:]
    X_1_test = dataset.iloc[35:50, :]
    X_2_train = dataset.iloc[50:85, :]
    X_2_test = dataset.iloc[85:100, :]
    X_3_train = dataset.iloc[100:135, :]
    X_3_test = dataset.iloc[135:150, :]
    train_set = train_set1 = pd.concat([X_1_train,X_2_train,X_3_train], ignore_index=True)
    test_set = pd.concat([ X_1_test, X_2_test, X_3_test], ignore_index=True)
    return train_set, test_set

#Softmax
"""
def softmax(x):
    pro1 = np.exp(x) / (np.sum(np.exp(x), axis=0))
    pro = np.amax(pro1)
    if(pro == 0):
        return [1,0,0]
    elif(pro==1):
        return [0,1,0]
    else:
        return [0,0,1]
"""
#Softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def get_class(x):
    a = np.exp(x) / np.sum(np.exp(x), axis=0)
    b = np.argmax(a)
    return b

def sigmod(x):
     return 1/(1+np.exp(-x))

sigmod_v = np.vectorize(sigmod)


def MLP(numHidden, numOutputs, train_set, eth, num_runs, test_set):

    # MSE
    ACC_Train = np.zeros((num_runs,1))
    ACC_Test = np.zeros((num_runs,1))
    #MSEdummy = np.zeros((num_runs,1))
    # Split the inputs and targets
    train_set_inputs = train_set.iloc[:, 0:2]
    train_set_inputs = train_set_inputs.values

    #add bias
    ih_bias = np.ones((numHidden,1))
    ho_bias =np.ones((numOutputs,1))

    #train_set_inputs = np.concatenate((train_set_inputs,np.ones((train_set_inputs.shape[0], 1))), axis=1)
    train_set_inputs = np.concatenate((np.ones((train_set_inputs.shape[0], 1)),train_set_inputs ), axis=1)
    #print(train_set_inputs)
    #print(train_set_inputs)
    train_set_targets = train_set.iloc[:, 2]
    r = train_set_targets.values
    new_tartgets = np.zeros((r.shape[0],numOutputs))
    #print(new_tartgets)
    for a in range(r.shape[0]):
        if(r[a] == 0):
            new_tartgets[a][0] = 1
        elif(r[a]==1):
            new_tartgets[a][1] = 1
        else:
            new_tartgets[a][2] =1
    #print(new_tartgets)

    numInput = train_set_inputs.shape[1]

    # Initialize the weights between inputs and hidden layers
    ih_weights = np.random.random((numInput, numHidden)) * 0.02 - 0.01
    #print(ih_weights)

    # Initialize the weights between hiddens and outputs
    ho_weights = np.random.random((numHidden + 1, numOutputs)) * 0.02 - 0.01
    for num in range(num_runs):
        # Accuary
        tc = 0
        fc = 0
        tct = 0
        fct = 0

        for i in range(train_set.shape[0]):
            x=train_set_inputs[i,:]
            yih = np.dot(x,ih_weights)
            #print(yih)
            #yih = softmax(yih)
            yih = sigmod_v(yih)
            #yih = np.vectorize()
            #print(yih)
            yih_bias = np.insert(yih,0,1)

            yho = np.dot(yih_bias, ho_weights)
            #print(yho)
            #Calculate the MSE
            pre_class = get_class(yho)

            #MSE_Train[num] = MSE_Train[num] + (r[i]-pre_class)**2
            #Get the accuary
            if (pre_class == r[i]):
                tc = tc + 1
            else:
                fc = fc + 1


            yho = softmax(yho)
            #print(yih)
            #print(yho)
            #Calculate the MSE
            #error = r[i] - get_class(yho)

            #Calculate the loss
            output_error = new_tartgets[i] - yho

            #print(new_tartgets[i])
            #print(yho)
            #print(output_error)

            #Calculate the hidden loss
            dho = np.ones((yih_bias.shape[0],output_error.shape[0]))
            #print(yih_bias.shape[0],output_error.shape[0])
            #dho = eth*(np.dot((np.transpose(output_error)),yih_bias))

            #Calculate the delta loss weight between hidden and output
            for a in range(5):
                for b in range(3):
                    dho[a, b] = output_error[b]*yih_bias[a]
            #print(dho)
            #Calculate the delta loss weight beteen input and hidden
            hidden_error = np.dot(ho_weights, output_error)
            #print(hidden_error)

            hidden_error_input = hidden_error[1:5]
            #print(hidden_error_input)

            # hidden * (1-hidden)
            #print(yih)
            #dsig = 0
            #for a in range(yih.shape[0]):
             #   dsig = dsig + yih[a] *(1-yih[a])
            dih = np.ones((numInput, numHidden))
            for a in range(numInput):
                for b in range(numHidden):
                    dih[a, b] = hidden_error[b] * x[a] *yih[b] *(1-yih[b])


            #UPDATE THE weights
            #print(ih_weights.shape,dih.shape)

            ih_weights = ih_weights + eth*dih
            ho_weights = ho_weights + eth*dho
        #MSE_Train[num] = float(MSE_Train[num]/(train_set.shape[0]))
        ACC_Train[num] = float(tc/ (tc+fc))


        x = test_set.iloc[:, 0:2]
        x = x.values
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        y = test_set.iloc[:,2]
        y = y.values
        #print(y)
        xh = np.dot(x,ih_weights)
        xh = sigmod_v(xh)
        xh = np.concatenate((np.ones((xh.shape[0], 1)), xh), axis=1)
        #xh_bias = np.insert(xh, 0, 1)
        ho = np.dot(xh, ho_weights)
        final_class = np.zeros((test_set.shape[0],1))

        for i in range(test_set.shape[0]):
            final_class[i] = get_class(ho[i])
            pre_cla = get_class(ho[i])
            #print(pre_cla)
            #print(pre_cla, y[i])
            #MSE_Test[num] = MSE_Test[num] + (y[i] == pre_cla)
            #print(y[i])
            if(pre_cla == y[i]):
                tct = tct + 1
            else:
                fct = fct +1


        ACC_Test[num] = float(tct/ (tct+fct))
        #MSEdummy[num] = np.mean((y-np.mean(r))**2)

    return ACC_Train, ACC_Test

def plotResult(ACC_Train, ACC_Test,num_runs,eth):
    num_runs= np.arange(0,num_runs)
    MSEdummy = np.full((190,1), 0.3)

    plt.plot(num_runs,ACC_Train , 'b', num_runs, ACC_Test, 'r', num_runs, MSEdummy, 'g')

    plt.show()










































"""
#Softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def MLP(numHidden, train_set, eth):
    #initialize the bias
    #ih_bias = [[1]* numHidden]*train_set.shape[0]
    #print(ih_bias)

    # Split the inputs and targets
    train_set_inputs = train_set.iloc[:, 0:2]
    train_set_inputs = train_set_inputs.values
    #Add the bias into inputs
    train_set_inputs = np.concatenate((np.ones((train_set_inputs.shape[0], 1)), train_set_inputs), axis=1)
    #print(train_set_inputs)
    train_set_targets = train_set.iloc[:, 2]
    train_set_targets = train_set_targets.values
    numInput = train_set_inputs.shape[1]
    #print(numInput)

    # Initialize the weights between inputs and hidden layers
    ih_weights = np.random.random((numInput,numHidden))*2 -1

    #Get the dot product of inputs and hidden layers
    #hidden_act = np.dot(train_set_inputs, ih_weights)
    #print(hidden_act)
    hidden_act = np.dot(train_set_inputs, ih_weights)
    #print(hidden_act)

    #Add bias into the product
    hidden_act = np.concatenate((np.ones((hidden_act.shape[0], 1)), hidden_act), axis=1)
    #print(hidden_act)
    #Initialize the weights between hiddens and outputs
    ho_weights = np.random.random((numHidden+1,3)) *2 - 1

    #get the dot product between hidden)act and ho_weights
    #output_activations = np.dot(hidden_act, ho_weights)
    #print(output_activations)
    output_activations = np.dot(hidden_act, ho_weights)
    #print(output_activations)

    #Pass to softmax
    #softmax_outputs = [[0]*3]*output_activations.shape[0]
    softmax_outputs = np.ones((105,3))
    for i in range(output_activations.shape[0]):
        softmax_outputs[i] = softmax(output_activations[i,:])
        #print(softmax_outputs[i])
    #print(softmax_outputs)

    mlp_results = np.argmax(softmax_outputs,axis =1)
    #print(mlp_results)

    #calculate the error
    error = train_set_targets - mlp_results
    print(np.dot(error, train_set_inputs))

"""

















main()