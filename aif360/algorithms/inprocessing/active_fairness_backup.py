"""
Active Fairness Run through questions
"""

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import _SigmoidCalibration 
from sklearn.isotonic import IsotonicRegression
from joblib import Parallel, delayed
import pathos.multiprocessing as multiprocessing
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import numpy as np
from collections import Counter
import numpy as np
import pandas as pd
import time
import random
from copy import deepcopy


class ActiveFairness(object):
    def __init__(self, 
                dataset_train, dataset_test, 
                clf,
                privileged_groups,
                unprivileged_groups,
                sensitive_features = [],
                target_label = [], 
                print_baselines=True):
        '''
        An object for running active fairness algorithm
        Input: 
            X_tr: features in training data, shape = [ N_instance, N_feature ], dtype = pd.Frame
            y_tr: outcome(classification) in training data
            X_te: features in testing data, shape = [ N_instance, N_feature ], dtype = pd.Frame
            y_te: outcome(classification) in testing data
            clf:  RF model
            dataset_name: name of the dataset
            print_baselines: whether to print baseline
        '''

        assert len(target_label) == 1, print("Error in ActiveFairness, length of target_label not defined")
        train = dataset_train.features
        test = dataset_test.features
        complete_data = dataset_train.metadata['previous'][0]
        feature_name = pd.DataFrame(complete_data.feature_names)
        y_column_index = ~(feature_name.isin(sensitive_features + target_label).iloc[:, 0])
        self.target_label = target_label
        self.sensitive_features = sensitive_features
        self.privileged_groups = privileged_groups
        self.unprivileged_groups = unprivileged_groups
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.X_tr = pd.DataFrame(train[:, y_column_index])
        self.y_tr = pd.DataFrame(self.dataset_train.labels[:, 0]).iloc[:, 0]
        self.X_te = pd.DataFrame(test[:, y_column_index])
        self.y_te = pd.DataFrame(self.dataset_test.labels[:, 0]).iloc[:, 0]

        # self.X_tr = train.loc[:, ~(train.columns.isin(sensitive_features + target_label))]
        # self.y_tr = train[target_label[0]]
        # self.X_te = test.loc[:, ~(train.columns.isin(sensitive_features + target_label))]
        # self.y_te = test[target_label[0]]

        self.clf = clf

    def fit(self): 
        # This is a temporary implementation
        self.clf = self.clf.fit(self.X_tr, self.y_tr)
        self.features_by_importance = self.clf.feature_importances_.argsort()[::-1] # get the importance of features based on trained RF
        self.all_features = list(range(self.X_te.shape[1]))

        # if print_baselines:
        #     y_te_hat = self.clf.predict(X_te)
        #     print("Accuracy: ", round(accuracy_score(y_te, y_te_hat) * 100, 2), "%")
        #     print("Classification report: \n", classification_report(y_te, y_te_hat, target_names=['not poor', 'poor']))
    
    def predict(self, train):
        if train == True:
            Y_tr_predict = self.clf.predict(self.X_tr)
            re_dataset_train = deepcopy(self.dataset_train)
            # re_dataset_train.metadata["params"]["df"][self.target_label[0]] = Y_tr_predict
            re_dataset_train.labels = Y_tr_predict
            return re_dataset_train
        else:
            Y_te_predict = self.clf.predict(self.X_te)
            re_dataset_test = deepcopy(self.dataset_test)
            # re_dataset_test.metadata["params"]["df"][self.target_label[0]] = Y_te_predict
            re_dataset_test.labels = Y_te_predict
            return re_dataset_test

    def run_algo_in_parallel(self, new_feat_mode,
                            verbose=1, 
                            plot_any=False, 
                            batch_size=512, 
                            nr_of_batches=100,
                             n_jobs=-1, 
                            save_to_file=True,
                            run_on_training=False,
                            save_folder=''):

        assert (len(save_folder) == 0) or (save_to_file)
        
        
        X_tr = self.X_tr
        y_tr = self.y_tr
        if run_on_training:
            X_te = self.X_tr
            y_te = self.y_tr
        else:
            X_te = self.X_te
            y_te = self.y_te
        clf = self.clf 
        all_features = self.all_features
        features_by_importance = self.features_by_importance
        start_time2 = time.time()

        results = []
        for ii in range(nr_of_batches):
            start = ii*batch_size
            end = min(X_te.shape[0] ,(ii+1) * batch_size)

            if start >= X_te.shape[0]:
                break

            # results_one = [run_per_test_case(int(i), X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance) for i in list(np.arange(start,end,))]
            print('START',start, 'END', end)


            results_one = []
            t_spend = []
            for i in list(np.arange(start,end)):
                time1 = time.time()
                results_one.append(run_per_test_case(i, X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance))
                t_spend.append(time.time() - time1)
            print("t_spend", t_spend)
            exit(1)
            # results_one = [run_per_test_case(i, X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance) for i in list(np.arange(start,end))]
            # results_one = Parallel(n_jobs=n_jobs, max_nbytes=None)(delayed(run_per_test_case)(i, X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance) for i in list(np.arange(start,end)))
            results.extend(results_one)

            ser_p = [pd.Series(results[i]['p_list'], name=results[i]['index']) for i in range(len(results))]
            df_p = pd.concat(ser_p,axis=1).transpose()
            df_p = (1-df_p) #correcting because somehow the p's are inversed

            ser_qa = [pd.Series(results[i]['qa'], name=results[i]['index']) for i in range(len(results))]
            df_qa = pd.concat(ser_qa,axis=1).transpose()

            ser_y = [pd.Series(results[i]['y_te'], name=results[i]['index']) for i in range(len(results))]
            df_y = pd.concat(ser_y,axis=1).transpose()
            
            ser_mc = [pd.Series(results[i]['max_conf'], name=results[i]['index']) for i in range(len(results))]
            df_mc = pd.concat(ser_mc,axis=1).transpose()

            df_X = pd.concat([results[i]['X_te'] for i in range(len(results))],axis=1).transpose()
            # print("PNKIE\nPNKIE\nPNKIE\nPNKIE\nPNKIE\n")
            if save_to_file:
                # print("dfslkjDDD\ndfslkjDDD\ndfslkjDDD\ndfslkjDDD\ndfslkjDDD\n")
                df_p.to_csv('{}/{}_dataframe_p_{}.csv'.format(save_folder,new_feat_mode,ii))
                df_qa.to_csv('{}/{}_dataframe_qa_{}.csv'.format(save_folder,new_feat_mode,ii))
                df_y.to_csv('{}/{}_dataframe_y_test_{}.csv'.format(save_folder,new_feat_mode,ii))
                df_mc.to_csv('{}/{}_dataframe_mc_{}.csv'.format(save_folder,new_feat_mode,ii))
                df_X.to_csv('{}/{}_dataframe_X_test_{}.csv'.format(save_folder,new_feat_mode,ii))

        return df_p, df_qa, df_y, df_mc, df_X # What does this part mean?

def run_per_test_case(test_case_id, X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance):
    start_time_00 = time.time()
    if verbose >= 1:
        if test_case_id % 1 == 0:
            print('Test case', test_case_id,"of",len(y_te))
            print('Time passed', time.time()-start_time2)
            print('Mode', new_feat_mode)
            print()

    features = deepcopy(all_features)
    test_example_full = X_te.iloc[test_case_id, :].values.astype(float)
    test_example = test_example_full * np.nan # Initialized as all nan
    time_stop1 = time.time()

    max_conf = clf.predict_proba(test_example_full.reshape(1,-1))[0][0]
    p = [] 
    question_asked = []
    times = []
    time_stop2 = time.time()

    # Ask questions until we hit the threshold or reach end of feature vector
    for question_i in range(len(features)):
        start_time = time.time()
        p_dict, p_cur = calcPValuesPerTree(test_example, clf)
        p.append(p_cur)

        # Choose one way to select the next feature in query
        if new_feat_mode == 'random':
            new_feature = random.sample(features,1)[0]
        elif new_feat_mode == 'feat-imp':
            new_feature = features_by_importance[question_i]
        elif new_feat_mode == 'ask-town':
            new_feature = getTheNextBestFeature(clf, features,test_example, p_cur, clf, absolutes_on=False)
        elif new_feat_mode == 'abs-agg':
            new_feature = getTheNextBestFeature(clf, features,test_example, p_cur, clf)
        else:
            raise Exception('mode has not been implemented')
        features.remove(new_feature)
        question_asked.append(new_feature)
        test_example[new_feature] = test_example_full[new_feature]
        times.append(time.time() - start_time)

        if verbose >= 3:
            print()  
            print('Test case', test_case_id,"of",len(y_tr), 'index', X_tr.index[test_case_id])
            print('Time passed', time.time()-start_time2)
            print("Nr of questions asked : ", question_i)
            print("P before classification : ", p[-1])
            print("feature asked : ", list(X_tr)[new_feature])
            print("feature number asked : ", new_feature)
            print("max conf", max_conf)
            print("Actual Conf after : ", calcPValuesPerTree(test_example, clf)[1])
    time_stop3 = time.time()

    if verbose >= 2:
        print("Test example's true label: ", y_te.iloc[test_case_id])
        print("Prevalence of class 0 in y_test: ", y_te.mean())

        # plot per test case
        if False:
            fig, ax1 = plt.subplots()
            ax1.set_title(str(test_case_id) + "|" + str(y_te.iloc[test_case_id]))
            ax1.plot(p, "gd-", label='probability of class 0')
            ax1.set_ylabel('probability of class 0')
            ax1.set_ylim([0, 1])
            ax2 = ax1.twinx()
            ax2.plot(times, 'bd-', label='computation time')
            ax2.set_ylabel('time')
            ax1.set_xlabel("Questions Asked")
            ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1)
            ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2)
            plt.show()
    # print("T-lap 0", time_stop1 - start_time_00)
    # print("T-lap 1", time_stop2 - time_stop1)
    # print("T-lap 2", time_stop3 - time_stop2)

    return {'X_te':X_te.iloc[test_case_id,:],'y_te':y_te.iloc[test_case_id],'max_conf':max_conf,'index':X_te.index[test_case_id],'p_list':p,'qa':question_asked}

def ClassifyWithPartialFeatures(sampleData,tree):
    rootNode = 0
    queue = [rootNode]
    
    children_left = tree.children_left
    children_right = tree.children_right

    threshold = tree.threshold
    feature = tree.feature
    values = tree.value
    weighted_samples = tree.weighted_n_node_samples
    
    total_w = 0
    total_probs = np.zeros_like(values[0])
    
    while queue:  # core tree search
        currentNode = queue.pop(0)

        if feature[currentNode] == -2:
            total_probs += values[currentNode]  # exits loop
            total_w += weighted_samples[currentNode]
            continue
        nodeThreshold = threshold[currentNode]
        nodeFeature = feature[currentNode]

        if np.isnan(sampleData[nodeFeature]):
            queue.append(children_left[currentNode])
            queue.append(children_right[currentNode])

        else:
            if sampleData[nodeFeature] <= nodeThreshold:
                queue.append(children_left[currentNode])
            else:
                queue.append(children_right[currentNode])

    norm_w = total_w / weighted_samples[feature == -2].sum()
    norm_probs = total_probs[0]/total_probs[0].sum()          
    
    return norm_probs[0], norm_w  # also return the weight

def getTheNextBestFeature(clf, features,test_example, p_cur, absolutes_on=True):
    # cfl = self.clf
    number_of_trees = len(clf.estimators_)
    next_feature_array = np.zeros(len(features))

    for feat_i, feature in enumerate(features):
        for tree_id in range(number_of_trees):
            tree = clf.estimators_[tree_id].tree_

            if feature in tree.feature:
                l = list(set(tree.threshold[tree.feature == feature] ))
                test_values = [l[0]-1e-3] + [(a + b) / 2. for a, b in zip(l, l[1:])] + [l[-1]+1e-3]
                conf_list = []
                w_list = []
                for value in test_values:
                    test_example_temp = test_example.copy()
                    test_example_temp[feature] = value
                    
                    p,w = ClassifyWithPartialFeatures(test_example_temp, tree)
                    if absolutes_on:
                        conf_list.append(np.abs(p-p_cur))
                    else:
                        conf_list.append(p)
                    w_list.append(w)

                next_feature_array[feat_i] += sum([x*y for x,y in zip(conf_list,w_list)]) / sum(w_list)
            else:
                if (not absolutes_on):
                    next_feature_array[feat_i] += p_cur
    
    if absolutes_on:
        return features[np.argmax(next_feature_array)]
    else:
        next_feature_array = next_feature_array / number_of_trees 
        return features[np.argmax(np.abs(next_feature_array-p_cur))]

def calcPValuesPerTree(test_example, clf):
    p_list = [ClassifyWithPartialFeatures(test_example,value.tree_)[0] for t_id,value in enumerate(clf.estimators_)]
    return p_list,np.mean(p_list)

class calibration(object):
    def __init__(self,method= 'sigmoid'):
        self.method = method
        
    def fit(self, p_input, y):
        if self.method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        elif self.method == 'sigmoid':
            calibrator = _SigmoidCalibration()
        calibrator.fit(p_input, y)
        if self.method == 'sigmoid':
            self.a = calibrator.a_
            self.b = calibrator.b_
        self.calibrator = calibrator
        
        return self 
    
    def predict(self, p_input):
        return self.calibrator.predict(p_input)

def calibrate_probs(df_p_train, df_p_test, y_train, y_test, group_train, group_test, cal_mode = 'sigmoid', calibration_type='per_group', print_random=True):
    '''
    df_p_train: the probability when having certain amount of features in queries(training data), shape = [Number of examples, Number of features asked]
    df_p_test: the probability when having certain amount of features in queries(testing data), shape = [Number of examples, Number of features asked]
    y_train: the actual label in the training dataset, shape = [Number of examples, 1]
    y_test: the actual label in the testing dataset, shape = [Number of examples, 1]
    group_train: the sensitive value in training dataset, type: Series
    group_test: the sensitive value in testing dataset, type: Series
    cal_mode: the model for calibration
    calibration_type: the type for calibration (Only group level calibration supported so far)
    print_random: print signal
    '''

    df_p_test.columns = [str(i) for i in df_p_test.columns]
    df_p_train.columns = [str(i) for i in df_p_train.columns]

    df_p_cal_test = df_p_test.copy()
    if calibration_type == 'per_group':
        for q_i in range(df_p_train.shape[1]):
            calibrator_per_group = {}

            for group in group_train.unique():
                X = df_p_train.loc[group_train == group,str(q_i)]
                Y = y_train.loc[group_train == group,0]

                calibrator_per_group[group] = calibration(cal_mode).fit(
                    X,Y) if len(Y) != 0 else None

            #TODO - speed up this part of the code
            for ii, (p_old, group, y_value) in enumerate(zip(df_p_test[str(q_i)],group_test,y_test.loc[:,0])): 
                df_p_cal_test[str(q_i)].iloc[ii]  = calibrator_per_group[group].predict(pd.Series(p_old))[0] if calibrator_per_group[group] != None else p_old
                # if print_random and (ii % 1000) ==0:
                #     print(q_i,ii,group, p_old,df_p_cal_test[str(q_i)].iloc[ii],y_value)
    else:
        raise ValueError('Calibration type {} is not yet supported'.format(calibration_type))
    return df_p_cal_test
