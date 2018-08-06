import helper
import copy
import numpy as np
import os
import math

def fool_classifier(test_data): ## Please do not change the function defination...

    strategy_instance=helper.strategy()

    parameters={'kernel':'linear',
                'C':1.0,
                'gamma':'auto',
                'degree':3,
                'coef0':0.0}

    # generate training dictionary
    training_set = set()
    for line in strategy_instance.class0:
        for token in line:
            training_set.add(token)
    for line in strategy_instance.class1:
        for token in line:
            training_set.add(token)
    dictionary = {}
    for token in training_set:
        dictionary[token] = 0
    dictionary_index = list(dictionary.keys())

    # generate x_train and y_train
    x = []
    y = []


    for line in strategy_instance.class0:
        tmp_dic = copy.deepcopy(dictionary)
        for token in line:
            tmp_dic[token] += 1
        x.append(list(tmp_dic.values()))
        y.append(0)
    for line in strategy_instance.class1:
        tmp_dic = copy.deepcopy(dictionary)
        for token in line:
            tmp_dic[token] += 1
        x.append(list(tmp_dic.values()))
        y.append(1)
    x = np.array(x)
    #  vector of n_sample that contains the token
    ti = []
    for col in range(len(x[0])):
        count = 0
        for row in range(len(x)):
            if x[row][col] >= 1:
                count += 1
        ti.append(count)

    # compute tf-idf to generate x_train
    tf_vector = []
    idf_vector = []
    for i in range(len(x)):
        tf = []
        idf = []
        row_sum = sum(x[i])
        for j in range(len(x[0])):
            tf.append( x[i][j] / row_sum)
            idf.append(math.log(x.shape[0] / (ti[j] + 1) ,2))
        tf_vector.append(tf)
        idf_vector.append(idf)

    x_train = np.array(tf_vector) * np.array(idf_vector)
    y_train = np.array(y)

    # training
    clf = strategy_instance.train_svm(parameters,x_train,y_train)

    #get the weights of features and sort them
    support_vector = clf.coef_[0]
    weighted_dictionary = copy.deepcopy(dictionary)
    for i in range(len(dictionary_index)):
        weighted_dictionary[dictionary_index[i]] = support_vector[i]
    sorted_weight = sorted(weighted_dictionary.items(),key = lambda x:x[1],reverse = False)
    
    ## modify test.txt based on sorted_weight
    with open(test_data,'r') as f:
        data=[line.strip().split(' ') for line in f]
    for n_line in range(len(data)):
        count = 0
        while count < 20:
            for i in range(len(sorted_weight)):
                if sorted_weight[-i-1][0] in data[n_line]:
                    tmp = [word for word in data[n_line] if word != sorted_weight[-i-1][0]]
                    data[n_line] = tmp
                    count += 1
                    break
    # for n_line in range(len(data)):
    #     count = 0
    #     while count < 10:
    #         for i in range(len(sorted_weight)):
    #             if sorted_weight[i][0] not in data[n_line]:
    #                 data[n_line] = data[n_line] + [sorted_weight[i][0]]
    #                 count += 1
    #                 break
    ## Write out the modified file, i.e., 'modified_data.txt' in Present Working Directory
    modified_data='./modified_data.txt'
    path = os.getcwd()
    os.chdir(path)
    with open(modified_data,"w") as f:
        for line in data:
            f.write(" ".join(sorted(line)) + "\n")

    # #### my test script########################
    # with open(modified_data,'r') as f:
    #     data=[line.strip().split(' ') for line in f]
    # x = []
    # for line in data:
    #     tmp_dic = copy.deepcopy(dictionary)
    #     for token in line:
    #         if token in tmp_dic:
    #             tmp_dic[token] += 1
    #     x.append(list(tmp_dic.values()))
    # x_test = np.array(x)
    # y_test = np.array([0 for _ in range(len(data))])
    # print(clf.predict(x_test))
    # print(clf.predict(x_test).shape)
    # print(clf.score(x_test,y_test))
    # ########################################




    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance

