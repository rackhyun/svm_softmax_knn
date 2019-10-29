import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg

# you can define/use whatever functions to implememt

# Problem 1.
# cross entropy loss
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    loss = 0.0
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1,n))

    #  this will give you a score matrix s of size (num_class)-by-(n)
    #  the i-th column vector of s will be
    #  the score vector of size (num_class)-by-1, for the i-th input data point # performing s=Wx+b
    s = W @ x + b # s : num_class * n matrix

    s = np.exp(s)
    s /= np.sum(s, axis=0) # divide by sum

    for i in range(n):
        loss -= np.log(s[y[i]][i])

    return loss
   #  for i in range(n):
   #      calc score
   #     s = W @ x[i].reshape((-1, 1)) + b
   #      exp
   #     s = np.exp(s)
   #      divide by sum
   #     s /= np.sum(s)
   #
   #     loss -= np.log(s[y[i]])
   #      print("loss :", np.log(s[y[i]]))
   #
   #  print('s : ', s)
   #
   # return loss


# Problem 2.
# svm loss calculation
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    # implement your function here
    # return SVM loss
    loss = 0.0
    # 일단 Wb를 W와 b로 나누자
    Wb = np.reshape(Wb, (-1, 1))
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    b = Wb[-num_class:]
    x = np.reshape(x.T, (-1, n))
    # W와 x를 내적하고 b를 더해서 s를 만들고
    s = W @ x + b # s : num_class * n

    for i in range(n):
        #svm_i = W @ x[i].reshape((-1, 1)) + b
        #print('svm_i : ', svm_i)
        for j in range(num_class):
            #print("svm_i[%d] : %d , y= %d", j, svm_i[j], y[i])
            #print("loss [i][j] : ", max(svm_i[j] - svm_i[y[i]] + 1, 0))
            # if (j == y[i]):
            #    continue
            # else:
            #    loss += max(svm_i[j] - svm_i[y[i]] + 1, 0)
            loss += max(s[j][i] - s[y[i]][i] + 1, 0)

    # svm에서 y값과 차이를 계산하여 loss를 구하자
    # 자기자신은 생략해야 하는데 각 case에 1이 더해졌으니 트레이닝 데이터 만큼 loss에서 빼줘야
    return loss - n

# Problem 3.
# kNN classification
def knn_test(X_train, y_train, X_test, y_test, n_train_sample, n_test_sample, k):
    # implement your function here
    #return accuracy
    accuracy = 0

    # vector 간 거리 구함
    # dists 100 * 400
    dists = -2 * X_test @ X_train.T + np.sum(X_train ** 2, axis=1) + np.sum(X_test ** 2, axis=1)[:, np.newaxis]
    #print (len(dists[0]))

    for i in range(n_test_sample):
        # argsort로 정렬 index 구함
        sortdist = dists[i].argsort()
        cls = np.zeros(k)
        # cls에 상위 k개의 값을 넣는다.
        for j in range(k):
            cls[j] = y_train[sortdist[j]]
            #print ("cls[", j,"] = y_train[", sortdist[j],"] = ",  y_train[sortdist[j]])

        # scipy.stats.mode로 k개 중 많이 나오는 거 찾음
        m = stats.mode(cls)
        #print (cls)
        #print (cls, " ", y_train[sortdist[i]], " ", m[0])

        # 많이 나온거랑 y_train 값이랑 비교해서 맞으면 accruracy + 1
        accuracy += (m[0] == y_test[i]).astype('uint8')

    return  accuracy / n_test_sample



# now lets test the model for linear models, that is, SVM and softmax
# loss 를 구하는 함수
def linear_classifier_test(Wb, x_te, y_te, num_class, n_test):
    # 열벡터로 변환 ex) 2*3 -> 6*1 로
    Wb = np.reshape(Wb, (-1, 1))
    # input의 feature 수 구함
    dlen = len(x_te[0])
    # Wb의 뒤에서 부터 num_class 크기만큼 가지고 옴
    # a[start:stop:step] # start through not past stop, by step
    # a[-1]    # last item in the array
    # a[-2:]   # last two items in the array
    # a[:-2]   # everything except the last two items
    # 앞에 2*num_class는 w로 사용하고 뒤에 num_class는  bias로 사용 (b)
    b = Wb[-num_class:]
    #  앞에 dlen*num_class는 W로 사용하고
    # reshape -> Wb에서 (num_class*dlen)만큼 가지고 와서, num_class * dlen (4*2) matrix로 만
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0;

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)  # s (num_class * 1)에서 제일 큰 값을 가진 index구함
        accuracy = accuracy + (res == y_te[i]).astype('uint8')      # s의 index가 y와 같으면(true) 1 / 틀리면(false) 0

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=False, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'all'
#classifiers = 'knn'
#classifiers = 'svm'
#classifiers = 'softmax'


if classifiers == 'svm':

    print('training SVM classifier...')
    # 평균은 0, 표준편차 1인 정규 분포로 numclass * 2 + num_class 의 크기로 w0생성 (ex) num class 가 4면 12개 짜리, 3이면 9개
    # 앞에 2*num_class는 w로 사용하고 뒤에 num_class는  bias로 사용 (b)
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    # w0을 초기값으로 svm_loss함수를 최소화하는 w0 구함
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')
    # svm으로 구한 Wb로 정확도 구함
    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')

    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    # w0을 초기값으로 softmax_loss함수를 최소화하는 w0 구함
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')
    # softmax로 구한 wb로 정확도를 계산
    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'all':
    print('training softmax classifier...')

    # 평균은 0, 표준편차 1인 정규 분포로 numclass * 2 + num_class 의 크기로 w0생성 (ex) num class 가 4면 12개 짜리, 3이면 9개
    # 앞에 2*num_class는 w로 사용하고 뒤에 num_class는  bias로 사용 (b)
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))

    # === softmax ===#
    # w0을 초기값으로 softmax_loss함수를 최소화하는 w0 구함
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')
    # softmax로 구한 wb로 정확도를 계산
    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class, n_test) * 100, '%')

    #=== svm ===#
    print('training SVM classifier...')


    # w0을 초기값으로 svm_loss함수를 최소화하는 w0 구함
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')
    # svm으로 구한 Wb로 정확도 구함
    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class, n_test) * 100, '%')

    # === knn ===#
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k) * 100
          , '% for k value of ', k)

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)