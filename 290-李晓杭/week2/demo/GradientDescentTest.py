import matplotlib.pyplot as pyplot
import math
import sys



X=[x*0.05 for x in range(10)]
Y=[0.55*x**2+0.6*x+3 for x in X]


# X = [0.01 * x for x in range(100)]
# Y = [2*x**2 + 3*x + 4 for x in X]
lr=0.1
w1,w2,w3=1,1,1

def funx(x):
    y=w1*x**2+w2*x+w3
    return y


def loss_value(y_true,y_pred):
    return (y_true-y_pred) ** 2

def diff_y(y_true,y_pred):
   return y_true - y_pred



for i in range(1000):
   loss_values=0
   for  x,y_true in zip(X,Y):
       y_pred = funx(x)
       loss_values +=loss_value(y_true,y_pred)

       grad_w1 = 2 * (y_pred - y_true) * x ** 2
       grad_w2 = 2 * (y_pred - y_true) * x
       grad_w3 = 2 * (y_pred - y_true)
       # 权重更新
       w1 = w1 - lr * grad_w1  # sgd
       w2 = w2 - lr * grad_w2
       w3 = w3 - lr * grad_w3

   loss_values /= len(X)
   print("第%d轮， loss %f" % (i, loss_values))
   if loss_values < 0.00001:
       break


print(f"训练后权重:w1:{w1} w2:{w2} w3:{w3}")
