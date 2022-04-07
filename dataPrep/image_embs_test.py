import json
import numpy as np
import pickle


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pickle

CONFIG_FILE_NAME = 'config.json'
 
if __name__ == '__main__':
    # read configuration
    with open(CONFIG_FILE_NAME) as config_file:
        config = json.load(config_file)

    
    # load data
    x = []
    for i in range(1,6):
        file_path = config['DATASET']["imgs_ftrs_path"]+"\\image_embs_omp1t_batch"+str(i)+".pkl"
        with open(file_path, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            if(len(x)==0):
                x = d
            else:
                x = np.concatenate((x,d))

    x = np.float64(x)

    path_labels = config['DATASET']['imgs_path']+"\\train_labels.pkl"
    with open(path_labels, 'rb') as f:
        y = pickle.load(f, encoding='bytes')
    '''
    import h5py
    with h5py.File('C:\\Users\\Emilija\\Desktop\\dipl\\dataset\\imageData\\features_git\cifar10\\train.mat', 'r') as f:
        x = np.array(f['trainX'])
        y = np.array(f['trainY']).T
    '''

    # split to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    
    del(x) #conserve mem
    del(y)
    
    # train classifier with feature normalization
    clf = make_pipeline(StandardScaler(copy=False), LinearSVC(random_state=42,penalty='l2', loss='hinge', C=0.001, tol=1e-2, dual=True, max_iter=500, verbose=1),verbose=True)    
    

    #clf = make_pipeline(StandardScaler(copy=False), SGDClassifier(penalty='l2', loss='hinge', learning_rate='adaptive', eta0=5, early_stopping = False, 
    #    alpha=0.04, tol=1e-3, fit_intercept = True, random_state=42, max_iter=1000, verbose=0),verbose=True)    
    
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(classification_report(y_true=y_train,y_pred=y_pred))
    print("train acc:",accuracy)

    #y_pred = clf.predict(x_val)
    #accuracy = accuracy_score(y_val, y_pred)
    #print(classification_report(y_true=y_val,y_pred=y_pred))
    #print("val acc:",accuracy)
    
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_true=y_test,y_pred=y_pred))
    print("test acc:",accuracy)
    
    # save trained svm model
    with open("svm_image_emb_model_boljeod60.pkl", 'wb') as f:
        pickle.dump(clf,f)
