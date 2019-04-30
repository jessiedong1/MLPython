from numpy import *


def main():
    pro_A, pro_B, a_mean, b_mean, a_sd, b_sd, x = Get_Values()

    pro_xA = Cal_Pro(pro_A,a_mean,a_sd,x)
    pro_xB = Cal_Pro(pro_B, b_mean, b_sd, x)
    Result(pro_A,pro_B)

def Get_Values():
    # Clas  A
    pro_A = float(input("Prior probablity for class A: "))
    a_mean = float(input("Mean for class A: "))
    a_sd = float(input("Standard deviation for class A: "))

    #Class B
    pro_B = float(input("Prior probablity for class B: "))
    b_mean = float(input("Mean for class B: "))
    b_sd = float(input("Standard deviation for class B: "))

    x = float(input("Your test example: "))


    return pro_A,pro_B,a_mean,b_mean,a_sd,b_sd,x

def Cal_Pro(pro_A, a_mean, a_sd, x):

    var = a_sd**2
    x_1 = -(x-a_mean)**2
    #print(x_1)
    x_1_1 = x_1/(2*var)
    #print(x_1_1)
    x_1_2 = exp(x_1_1)
    #print(x_1_2)
    x_3 = 2*pi*var
    #print(x_3)
    x_2 = 1/sqrt(x_3)
    #print(x_2)
    pro_x_a = x_2*x_1_2*pro_A


    print(pro_x_a)

    return pro_x_a

def Result(a,b):
    if(a>b):
        print("This sample is Class A")

    else:
        print("This sample is from Class B")

main()



