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


class TreeNode:

    '''
     A node in the "featured tree"
    '''
    def __init__(self, threshold, dummy = False):
        '''
        threshold: The threshold of this node
        dummy: whether it's a fake node or not (The fake node can only be the root node of the tree)
        '''
        self.children_left = [] # nodes in its left (and of lower level in original tree)
        self.children_right = [] # nodes in its right (and of lower level in original tree)
        self.threshold = threshold 
        self.node_set = [set(), set()] # set of leaf nodes in its left and right, 
                                       # self.node_set[0] are the nodes in the left
                                       # self.node_set[1] are the nodes in the right
        self.dummy = dummy 


class TreeProcess:

    def __init__(self, tree, all_features):
        '''
        tree: the tree trained by random forest
        all_features: all possible features in this tree
        '''
        rootNode = 0
        node_trace = []
        self.children_left = tree.children_left
        self.children_right = tree.children_right
        self.threshold = tree.threshold
        self.feature = tree.feature
        self.values = tree.value
        self.weighted_samples = tree.weighted_n_node_samples
        # children_left, children_right, threshold, feature, values, weighted_samples used as a dict. Can provide corresponding value given an index of that node.
        
        self.total_leaf_id = set() # ids of all leaves in this tree
        self.feature2nodes = {} # dict, key is the name of features, value is the TreeNode object of the root for that 'feature tree'
        self.nodeid2TreeNode = {} # dict, key is the id of nodes in original tree, value is the TreeNode object corresponds to that node
        self.feature2threshold_list = {} # dict, key is name of features, value is a list of all thresholds for that feature
        self.featureAndthreshold2delete_set = {} # dict, key is name of features, value is another dict, with key as threshold value, and value as a set of leaf node ids to be delted
        self.tree_single_value_shape = np.shape(self.values[0]) # imitate the shape of 'self.values[0]'
        self.unique_feature = set() # total features exist in this tree (different from self.feature, which are features)

        if self.feature[rootNode] == -2:
            assert False, "The root of a tree is a leaf, please verify"

        for feature in all_features:
            # construct feature tree for all features
            
            queue = [rootNode]
            if feature == self.feature[rootNode]:
                # if the root node of original tree is of this feature, there is no need for a dummy
                queue = []
                self.nodeid2TreeNode[rootNode] = TreeNode(self.threshold[rootNode])
                self.feature2nodes[feature] = self.nodeid2TreeNode[rootNode]

                result_list = []
                left_node = self.children_left[rootNode]
                self.node_traverse(result_list, left_node, feature) # get all non-leaf nodes of this feature in the left sub-tree
                self.nodeid2TreeNode[rootNode].children_left = result_list

                result_list = []
                right_node = self.children_right[rootNode]
                self.node_traverse(result_list, right_node, feature) # get all non-leaf nodes of this feature in the right sub-tree
                self.nodeid2TreeNode[rootNode].children_right = result_list

                result_set = set()
                self.node_traverse_leaf(result_set, left_node) # get all leaf nodes it can reach in the left sub-tree
                self.nodeid2TreeNode[rootNode].node_set[0] = result_set

                result_set = set()
                self.node_traverse_leaf(result_set, right_node) # get all leaf nodes it can reach in the right sub-tree
                self.nodeid2TreeNode[rootNode].node_set[1] = result_set

                queue.append(left_node)
                queue.append(right_node)

            else:
                # if the root node of original tree is not of this feature, we need to have a dummy root for this feature tree
                self.feature2nodes[feature] = TreeNode(-1, True) # add a dummy root

                result_list = []
                left_node = self.children_left[rootNode]
                self.node_traverse(result_list, left_node, feature) # get all non-leaf nodes of this feature in the left sub-tree
                self.feature2nodes[feature].children_left = result_list
                
                result_list = []
                right_node = self.children_right[rootNode]
                self.node_traverse(result_list, right_node, feature)# get all non-leaf nodes of this feature in the right sub-tree
                self.feature2nodes[feature].children_right = result_list

            # find the target
            
            while queue:
                current_node = queue.pop(0)
                if feature == self.feature[current_node]:
                    # find a node of given feature
                    self.nodeid2TreeNode[current_node] = TreeNode(self.threshold[current_node])
                    
                    result_list = []
                    left_node = self.children_left[current_node]
                    self.node_traverse(result_list, left_node, feature) # get all non-leaf nodes of this feature in the left sub-tree
                    self.nodeid2TreeNode[current_node].children_left = result_list
                    
                    result_list = []
                    right_node = self.children_right[current_node]
                    self.node_traverse(result_list, right_node, feature) # get all non-leaf nodes of this feature in the right sub-tree
                    self.nodeid2TreeNode[current_node].children_right = result_list

                    result_set = set()
                    self.node_traverse_leaf(result_set, left_node) 
                    self.nodeid2TreeNode[current_node].node_set[0] = result_set # get all leaf nodes it can reach in the left sub-tree

                    result_set = set()
                    self.node_traverse_leaf(result_set, right_node)
                    self.nodeid2TreeNode[current_node].node_set[1] = result_set # get all leaf nodes it can reach in the right sub-tree

                if self.feature[current_node] != -2:
                    # if not the leaf
                    queue.append(self.children_left[current_node])
                    queue.append(self.children_right[current_node])

        for feature in all_features:
            threshold_set = set()
            queue = [self.feature2nodes[feature]] # get the root in feature tree
            while queue:
                currentNode = queue.pop(0)
                if currentNode.dummy != True:
                    threshold_set.add(currentNode.threshold)

                for node in currentNode.children_left:
                    queue.append(self.nodeid2TreeNode[node])
                
                for node in currentNode.children_right:
                    queue.append(self.nodeid2TreeNode[node])

            threshold_list = sorted(list(threshold_set)) # rank the list in increasing threshold
            self.feature2threshold_list[feature] = threshold_list
            self.featureAndthreshold2delete_set[feature] = {}

        for feature in self.feature2threshold_list.keys():
            l = len(self.feature2threshold_list[feature])
            for i in range(l):
                threshold = self.feature2threshold_list[feature][i]
                delete_set_equal_or_less = set() # the nodes to be deleted if equal or less than the threshold
                queue = [self.feature2nodes[feature]] # the root of feature tree
                while queue:
                    currentTreeNode = queue.pop(0)
                    if currentTreeNode.dummy == True:
                        for node in currentTreeNode.children_left:
                            queue.append(self.nodeid2TreeNode[node])
                        for node in currentTreeNode.children_right:
                            queue.append(self.nodeid2TreeNode[node])
                    else:
                        if threshold <= currentTreeNode.threshold:
                            # current value (threshold) is equal or less than threshold for this node, go to the left sub-tree for this node
                            for node in currentTreeNode.children_left:
                                queue.append(self.nodeid2TreeNode[node])
                            delete_set_equal_or_less |= currentTreeNode.node_set[1] # delete all leaf-nodes can be reached in the right sub-tree
                        else:
                            for node in currentTreeNode.children_right:
                                queue.append(self.nodeid2TreeNode[node])
                            delete_set_equal_or_less |= currentTreeNode.node_set[0]    

                dummy_threshold = threshold + 1e-10 # make the current threshold slightly higher than the actual threshold

                if i < l - 1:
                    # prevent possible bugs
                    assert dummy_threshold < self.feature2threshold_list[feature][i+1], "threshold error" 

                delete_set_larger = set() # the nodes to be deleted if equal or larger than the threshold
                queue = [self.feature2nodes[feature]] # the first one
                while queue:
                    currentTreeNode = queue.pop(0)
                    if currentTreeNode.dummy == True:
                        for node in currentTreeNode.children_left:
                            queue.append(self.nodeid2TreeNode[node])
                        for node in currentTreeNode.children_right:
                            queue.append(self.nodeid2TreeNode[node])
                    else:
                        if dummy_threshold <= currentTreeNode.threshold:
                            for node in currentTreeNode.children_left:
                                queue.append(self.nodeid2TreeNode[node])
                            delete_set_larger |= currentTreeNode.node_set[1]
                        else:
                            for node in currentTreeNode.children_right:
                                queue.append(self.nodeid2TreeNode[node])
                            delete_set_larger |= currentTreeNode.node_set[0]    

                self.featureAndthreshold2delete_set[feature][threshold] = [delete_set_equal_or_less, delete_set_larger]

        for feature in self.feature2threshold_list.keys():
            if len(self.feature2threshold_list[feature]) > 0:
                self.unique_feature.add(feature)

    def node_traverse_leaf(self,
                        result_set,
                        currentNode):

        nodeFeature = self.feature[currentNode]
        if nodeFeature == -2:
            result_set.add(currentNode)
            self.total_leaf_id.add(currentNode)
            return
        self.node_traverse_leaf(result_set, self.children_left[currentNode])
        self.node_traverse_leaf(result_set, self.children_right[currentNode])

    def node_traverse(self,
                    result_list,
                    currentNode,
                    feature_target):
        nodeFeature = self.feature[currentNode]
        if nodeFeature == feature_target:
            result_list.append(currentNode)
            return
        if nodeFeature == -2:
            return 
        self.node_traverse(result_list, self.children_left[currentNode], feature_target)
        self.node_traverse(result_list, self.children_right[currentNode], feature_target)





class ActiveFairness(object):
    def __init__(self, 
                dataset_train, dataset_test, 
                clf,
                privileged_groups,
                unprivileged_groups,
                sensitive_features = [],
                target_label = [], 
                print_baselines=True):


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
        self.clf = clf

    def fit(self): 
        # This is a temporary implementation
        self.clf = self.clf.fit(self.X_tr, self.y_tr)
        self.features_by_importance = self.clf.feature_importances_.argsort()[::-1] # get the importance of features based on trained RF
        self.all_features = list(range(self.X_te.shape[1]))

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
                            save_folder='',
                            static_threshold = -1):

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
        self.trees = [TreeProcess(value.tree_, self.all_features) for value in clf.estimators_]
        all_features = self.all_features

        features_by_importance = self.features_by_importance
        start_time2 = time.time()

        results = []
        for ii in range(nr_of_batches):
            start = ii*batch_size
            end = min(X_te.shape[0] ,(ii+1) * batch_size)

            if start >= X_te.shape[0]:
                break

            print('START',start, 'END', end)
            results_one = [run_per_test_case(i, X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance, self.trees, static_threshold) for i in list(np.arange(start,end))]
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

        if save_to_file:
            df_p.to_csv('{}/{}_dataframe_p_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_qa.to_csv('{}/{}_dataframe_qa_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_y.to_csv('{}/{}_dataframe_y_test_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_mc.to_csv('{}/{}_dataframe_mc_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_X.to_csv('{}/{}_dataframe_X_test_{}.csv'.format(save_folder,new_feat_mode,ii))

        return df_p, df_qa, df_y, df_mc, df_X # What does this part mean?

def run_per_test_case(test_case_id, X_tr, y_tr, X_te, y_te, verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance, forestProcess, static_threshold = -1):
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
    if new_feat_mode == 'feat-imp':
        assert static_threshold != -1, "Static threshold must be set if static limit is set"
        upper_bound = min(static_threshold, len(features))
    else:
    	upper_bound = len(features)
    time_stop2 = time.time()

    previous_answers = [tree.total_leaf_id.copy() for tree in forestProcess]
    p_cur = -1
    for question_i in range(upper_bound):

        if new_feat_mode == 'random':
            new_feature = random.sample(features,1)[0]
        elif new_feat_mode == 'feat-imp':
            new_feature = features_by_importance[question_i]
        elif new_feat_mode == 'ask-town':
            new_feature = getTheNextBestFeature(forestProcess, features, test_example, previous_answers, p_cur, absolutes_on=False)
        elif new_feat_mode == 'abs-agg':
            new_feature = getTheNextBestFeature(forestProcess, features, test_example, previous_answers, p_cur)
        else:
            raise Exception('mode has not been implemented')
        p_dict, p_cur = calcPValuesPerTree(test_example_full, forestProcess, previous_answers, new_feature)
        p.append(p_cur)

        features.remove(new_feature)
        question_asked.append(new_feature)
        test_example[new_feature] = test_example_full[new_feature]

        if verbose >= 3:
            print()  
            print('Test case', test_case_id,"of",len(y_tr), 'index', X_tr.index[test_case_id])
            print('Time passed', time.time()-start_time2)
            print("Nr of questions asked : ", question_i)
            print("P before classification : ", p[-1])
            print("feature asked : ", list(X_tr)[new_feature])
            print("feature number asked : ", new_feature)
            print("max conf", max_conf)
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
            # ax2.plot(times, 'bd-', label='computation time')
            ax2.set_ylabel('time')
            ax1.set_xlabel("Questions Asked")
            ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1)
            ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2)
            plt.show()

    # used to debug
    # print("T-lap 0", time_stop1 - start_time_00)
    # print("T-lap 1", time_stop2 - time_stop1)
    # print("T-lap 2", time_stop3 - time_stop2)
    # return_time.append(time_stop3 - time_stop2)
    
    return {'X_te':X_te.iloc[test_case_id,:],'y_te':y_te.iloc[test_case_id],'max_conf':max_conf,'index':X_te.index[test_case_id],'p_list':p,'qa':question_asked}




def ClassifyWithPartialFeatures(sampleData,tree, previous_answers, new_feature):
    
    value = sampleData[new_feature]
    l = len(tree.feature2threshold_list[new_feature])
    if l > 0:
        Larger_than_all = True
        for i in range(l):
            if value <= tree.feature2threshold_list[new_feature][i]:
                Larger_than_all = False
                if i == 0:
                    previous_answers -= tree.featureAndthreshold2delete_set[new_feature][tree.feature2threshold_list[new_feature][i]][0]
                else:
                    previous_answers -= tree.featureAndthreshold2delete_set[new_feature][tree.feature2threshold_list[new_feature][i - 1]][1]
                    previous_answers -= tree.featureAndthreshold2delete_set[new_feature][tree.feature2threshold_list[new_feature][i]][0]
                break

        if Larger_than_all == True:
            previous_answers -= tree.featureAndthreshold2delete_set[new_feature][tree.feature2threshold_list[new_feature][l - 1]][1]

    total_w = 0
    total_probs = np.zeros(tree.tree_single_value_shape, dtype = np.float32)

    for node in previous_answers:
        total_probs += tree.values[node]
        total_w += tree.weighted_samples[node]

    norm_w = total_w / tree.weighted_samples[tree.feature == -2].sum()
    norm_probs = total_probs[0]/total_probs[0].sum()

    return norm_probs[0], norm_w  # also return the weight

def getTheNextBestFeature(forest, features, test_example, previous_answers, p_cur = -1, absolutes_on=True):
    number_of_trees = len(forest)
    next_feature_array = np.zeros(len(features))

    for feat_i, feature in enumerate(features):
        for tree_id in range(number_of_trees):
            tree = forest[tree_id]
            if feature in tree.unique_feature:
                l = tree.feature2threshold_list[feature]
                test_values = [l[0]-1e-3] + [(a + b) / 2. for a, b in zip(l, l[1:])] + [l[-1]+1e-3]
                conf_list = []
                w_list = []
                for value in test_values:
                    input_answer = previous_answers[tree_id].copy()
                    test_example_temp = test_example.copy()
                    test_example_temp[feature] = value
                    p,w = ClassifyWithPartialFeatures(test_example_temp, tree, input_answer, feature)
                    if absolutes_on and p_cur != -1:
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

def calcPValuesPerTree(test_example, forest, previous_answers, new_feature):

    p_list = [ClassifyWithPartialFeatures(test_example,tree, previous_answers[i], new_feature)[0] for i, tree in enumerate(forest)]
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

            for ii, (p_old, group, y_value) in enumerate(zip(df_p_test[str(q_i)],group_test,y_test.loc[:,0])): 
                df_p_cal_test[str(q_i)].iloc[ii]  = calibrator_per_group[group].predict(pd.Series(p_old))[0] if calibrator_per_group[group] != None else p_old

    else:
        raise ValueError('Calibration type {} is not yet supported'.format(calibration_type))
    return df_p_cal_test
