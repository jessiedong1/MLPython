from numpy import *

def main():
    eta = 0.25
    inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = array([[0], [1], [1], [1]])
    inputs = concatenate((-ones((inputs.shape[0], 1)), inputs), axis=1)
    Preceptron(inputs,targets,eta,5)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def LinearPerceptron(numInput, numOutput, inputs_data, label_data, eta, num_runs):
    weights = random.rand(numInput, numOutput) * 0.1 - 0.05
    for num in range(num_runs):
        pre_output = dot(inputs_data, weights)
        pre_output = pre_output.softmax(pre_output)




def Preceptron(inputs, targets, eta, n):
    #Inputs
    #inputs = array([-1,0,0],[-1,0,1],[-1, 1,0],[-1,1,1])
    #weights = random.rand(3,1)*0.1-0.05
    #weights = array([[-0.05], [-0.02], [0.02]])

    weights = random.rand(inputs.shape[1],1)*0.1-0.05

    for n in range(n):
        print("Round ", n)
        activations = dot(inputs, weights)
        #print(activations)
        activations = where(activations>0,1,0)
        print(activations)
        weights += eta*dot(transpose(inputs), targets-activations)
        print(weights)
        print()
    activations = dot(inputs, weights)
    #print(activations)
    activations = where(activations>0,1,0)
    print("Final: ")
    print(activations)


main()




