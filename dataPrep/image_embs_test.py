import numpy as np
import os

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import  data_utils 

if __name__ == '__main__':
    # read configuration
    config = data_utils.load_config()

    # load data
    x = data_utils.load_image_embeddings(config['DATASET']['IMAGES']["imgs_ftrs_path"], config['DATASET']['IMAGES']['imgs_ftrs_filename'], config['DATASET']['IMAGES']['imgs_batch_num'])

    path_labels = os.path.join(config['DATASET']['imgs_path'],"train_labels.pkl")
    y = data_utils.load_data(path_labels)
    
    # split to train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    del(x) #conserve mem
    del(y)

    # scale train and test data
    clf=StandardScaler(copy=False)
    clf.fit(X=x_train)
    x_train=clf.transform(x_train)
    x_test = clf.transform(x_test)
    
    clf = SGDClassifier(penalty='l2', loss='hinge', learning_rate='adaptive', eta0=1, early_stopping = False, 
        alpha=0.005, tol=1e-4, fit_intercept = True, random_state=42, max_iter=1000, verbose=1) 

    # 86 train 76 test sa 0.005

    # train linear classifier
    clf.fit(x_train, y_train)

    # evaluate
    y_pred = clf.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(classification_report(y_true=y_train,y_pred=y_pred))
    print("train acc:",accuracy)
    
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(classification_report(y_true=y_test,y_pred=y_pred))
    print("test acc:",accuracy)
    
    # save trained svm model
    img_model_filename = "svm_image_emb_model.pkl"
    data_utils.save_data(img_model_filename, clf)
