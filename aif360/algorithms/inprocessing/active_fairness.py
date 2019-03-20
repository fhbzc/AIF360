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
        child_left_dict = {}
        child_right_dict = {}
        for i in range(len(self.children_left)):
            child_left_dict[i] = self.children_left[i]
        for i in range(len(self.children_right)):
            child_right_dict[i] = self.children_right[i]
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
            if l == 0:
                continue
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


                self.featureAndthreshold2delete_set[feature][threshold] = delete_set_equal_or_less


            delete_set_larger = set() # the nodes to be deleted if larger than the threshold
            queue = [self.feature2nodes[feature]] # the root of feature tree
            while queue:
                currentTreeNode = queue.pop(0)
                if currentTreeNode.dummy == True:
                    for node in currentTreeNode.children_left:
                        queue.append(self.nodeid2TreeNode[node])
                    for node in currentTreeNode.children_right:
                        queue.append(self.nodeid2TreeNode[node])
                else:
                    for node in currentTreeNode.children_right:
                        queue.append(self.nodeid2TreeNode[node])
                    delete_set_larger |= currentTreeNode.node_set[0]    

            self.featureAndthreshold2delete_set[feature][np.inf] = delete_set_larger

        for feature in self.feature2threshold_list.keys():
            if len(self.feature2threshold_list[feature]) > 0:
                self.unique_feature.add(feature)





    def node_traverse_leaf(self,
                        result_set,
                        currentNode):
        # get all leaf nodes which can be reached starting from one node
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
                sensitive_features = [],
                target_label = []):

        '''
        dataset_train: training dataset, type: MexicoDataset()
        dataset_test: testing dataset, type: MexicoDataset()
        clf: trained randomforest classifier
        sensitive_features: a list of sensitive features which should be removed when doing prediction
        target_label: a list of features whose values are to be predicted
        '''
        assert len(target_label) == 1, print("Error in ActiveFairness, length of target_label not defined")
        train = dataset_train.features
        complete_data = dataset_train.metadata['previous'][0]
        self.feature2columnmap = {}
        test = dataset_test.features
        feature_name = pd.DataFrame(complete_data.feature_names)
        y_column_index = ~(feature_name.isin(sensitive_features + target_label).iloc[:, 0])
        y_column_index_inverse = (feature_name.isin(sensitive_features + target_label).iloc[:, 0])
        index = 0
        for i in range(len(y_column_index_inverse)):
            if y_column_index_inverse.iloc[i] == True:
                self.feature2columnmap[complete_data.feature_names[i]] = index
                index += 1
        self.target_label = target_label
        self.sensitive_features = sensitive_features
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test

        self.X_tr_sensitiveAtarget = pd.DataFrame(train[:, y_column_index_inverse]) # the dataframe of all samples in training dataset which only keeps the non-sensitive and target features
        self.X_tr = pd.DataFrame(train[:, y_column_index])
        self.y_tr = pd.DataFrame(self.dataset_train.labels[:, 0]).iloc[:, 0]
        self.X_te_sensitiveAtarget = pd.DataFrame(test[:, y_column_index_inverse]) # the dataframe of all samples in testing dataset which only keeps the non-sensitive and target features
        self.X_te = pd.DataFrame(test[:, y_column_index])
        self.y_te = pd.DataFrame(self.dataset_test.labels[:, 0]).iloc[:, 0]
        self.clf = clf
        self.trees = []

    def fit(self): 
        # This is a temporary implementation
        self.clf = self.clf.fit(self.X_tr, self.y_tr)
        self.features_by_importance = self.clf.feature_importances_.argsort()[::-1] # get the importance of features based on trained RF
        self.all_features = list(range(self.X_te.shape[1]))

    def predict(self, train):
        if train == True:
            Y_tr_predict = self.clf.predict(self.X_tr)
            re_dataset_train = deepcopy(self.dataset_train)
            re_dataset_train.labels = Y_tr_predict
            return re_dataset_train
        else:
            Y_te_predict = self.clf.predict(self.X_te)
            re_dataset_test = deepcopy(self.dataset_test)
            re_dataset_test.labels = Y_te_predict
            return re_dataset_test




    # choose the appropriate number of features to ask for Group A and B
    def choose_appropriate_num_of_feature(self, privilige_feature, privilige_value, unprivilige_value, \
                                          total_budget, feat_method = 'feat-imp', run_on_training = False):
        num_of_priviledge = 0
        num_of_unpriviledge = 0
        dataset = self.X_te_sensitiveAtarget if run_on_training == False else self.X_tr_sensitiveAtarget
        featured_dataset = self.X_te if run_on_training == False else self.X_tr
        for i in range(len(dataset)):
            if dataset.iloc[i, self.feature2columnmap[privilige_feature]] == privilige_value:
                # priviledge class
                num_of_priviledge += 1
            else:
                assert dataset.iloc[i, self.feature2columnmap[privilige_feature]] == unprivilige_value, "Value incorrect!"
                num_of_unpriviledge += 1
        total_num = num_of_priviledge + num_of_unpriviledge
        current_num_of_feature_for_priviledge = 0
        current_num_of_feature_for_unpriviledge = 0
        budget_used = 0
        # batch_size = 500
        # nr_of_batches = total_num // batch_size + 2
        dataset_orig = self.dataset_test if run_on_training == False else self.dataset_train
        self.trees = [TreeProcess(value.tree_, self.all_features) for value in self.clf.estimators_]
        
        features_by_importance = self.features_by_importance
        last_add_privi = True
        result = np.zeros([len(dataset)], dtype = np.float32)
        priviledge_index = []
        unprivilege_index = []
        for i in range(len(dataset)):
            if dataset.iloc[i, self.feature2columnmap[privilige_feature]] == privilige_value:
                priviledge_index.append(i)
            else:
                unprivilege_index.append(i)
        less_than_pri = np.array(dataset_orig.labels[priviledge_index] <= 0.5, dtype = bool)[:, 0]
        less_than_unpri = np.array(dataset_orig.labels[unprivilege_index] <= 0.5, dtype = bool)[:, 0]

        previous_answers = [[tree.total_leaf_id.copy() for tree in self.trees] for i in range(len(dataset))]

        print("Start the process")

        while budget_used < total_budget:
            # FP_pri = 0
            # TN_pri = 0
            # FP_unpri = 0
            # TN_unpri = 0
            
            if current_num_of_feature_for_priviledge == 0:
                FP_pri = 1
                TN_pri = 0
            else:
                privi_predict_result = np.array(result[priviledge_index] > 0.5, dtype = bool)
                FP_pri = np.sum(privi_predict_result * less_than_pri)
                TN_pri = np.sum((1 - privi_predict_result) * less_than_pri)

            if current_num_of_feature_for_unpriviledge == 0:
                FP_unpri = 1
                TN_unpri = 0
            else:
                unprivi_predict_result = np.array(result[unprivilege_index] > 0.5, dtype = bool)
                FP_unpri = np.sum(unprivi_predict_result * less_than_unpri)
                TN_unpri = np.sum((1 - unprivi_predict_result) * less_than_unpri)


            # for i in range(len(dataset)):
            #     if dataset.iloc[i, self.feature2columnmap[privilige_feature]] == privilige_value:
            #         # priviledge class
            #         if dataset_orig.labels[i] <= 0.5:
            #             # actual negative
            #             if current_num_of_feature_for_priviledge == 0:
            #                 FP_pri += 1
            #             else:   
            #                 if result[i] > 0.5:
            #                     FP_pri += 1
            #                 else:
            #                     TN_pri += 1
            #     else:
            #         if dataset_orig.labels[i] <= 0.5:
            #             # actual negative
            #             if current_num_of_feature_for_unpriviledge == 0:
            #                 FP_unpri += 1
            #             else:
            #                 if result[i] > 0.5:
            #                     FP_unpri += 1
            #                 else:
            #                     TN_unpri += 1
            FPR_pri = FP_pri * 1.0 / (FP_pri + TN_pri)   
            FPR_unpri = FP_unpri * 1.0 / (FP_unpri + TN_unpri)   
            result[:] = 0
            if FPR_pri > FPR_unpri:
                current_num_of_feature_for_priviledge += 1
                last_add_privi = True
                budget_used += (num_of_priviledge* 1.0 / total_num)
            else:
                current_num_of_feature_for_unpriviledge += 1
                last_add_privi = False
                budget_used += (num_of_unpriviledge * 1.0 / total_num)
            print("budget_used", budget_used)
            print("FPR_pri", FPR_pri)
            print("FPR_unpri", FPR_unpri)
            print("FP_pri", FP_pri)
            print("TN_pri", TN_pri)
            print("FP_unpri", FP_unpri)
            print("TN_unpri", TN_unpri)

            features = deepcopy(self.all_features)
            
            for j in range(len(dataset)):
                test_example_full = featured_dataset.iloc[j, :].values.astype(float)
                if dataset.iloc[j, self.feature2columnmap[privilige_feature]] == privilige_value and last_add_privi == True:
                    # priviledge class
                    if feat_method == 'random':
                        new_feature = random.sample(features,1)[0]
                        features.remove(new_feature)
                    elif feat_method == 'feat-imp':
                        new_feature = features_by_importance[current_num_of_feature_for_priviledge]
                    elif feat_method == 'ask-town':
                        assert False, "Error 385, not supported yet"
                        new_feature = getTheNextBestFeature(self.trees, features, test_example, previous_answers, p_cur, absolutes_on=False)
                        features.remove(new_feature)
                    elif feat_method == 'abs-agg':
                        assert False, "Error 389, not supported yet"
                        new_feature = getTheNextBestFeature(self.trees, features, test_example, previous_answers, p_cur)
                        features.remove(new_feature)
                    else:
                        raise Exception('mode has not been implemented')
                    p_dict, p_cur = calcPValuesPerTree(test_example_full, self.trees, previous_answers[j], new_feature)
                    result[j] = 1 - p_cur # somehow inversed
                    
                elif dataset.iloc[j, self.feature2columnmap[privilige_feature]] != privilige_value and last_add_privi == False:
                    if feat_method == 'random':
                        new_feature = random.sample(features,1)[0]
                        features.remove(new_feature)
                    elif feat_method == 'feat-imp':
                        new_feature = features_by_importance[current_num_of_feature_for_unpriviledge]
                    elif feat_method == 'ask-town':
                        assert False, "Error 385, not supported yet"
                        new_feature = getTheNextBestFeature(self.trees, features, test_example, previous_answers, p_cur, absolutes_on=False)
                        features.remove(new_feature)
                    elif feat_method == 'abs-agg':
                        assert False, "Error 389, not supported yet"
                        new_feature = getTheNextBestFeature(self.trees, features, test_example, previous_answers, p_cur)
                        features.remove(new_feature)
                    else:
                        raise Exception('mode has not been implemented')
                    p_dict, p_cur = calcPValuesPerTree(test_example_full, self.trees, previous_answers[j], new_feature)
                    result[j] = 1 - p_cur # somehow inversed
                    



        return current_num_of_feature_for_priviledge, current_num_of_feature_for_unpriviledge



    def run_algo_in_parallel(self, new_feat_mode,
                            sensitive_name, 
                            privilige_variable_value,
                            unprivilige_variable_value,
                            pri_num_feature_fetched,
                            un_pri_num_feature_fetched,
                            verbose=1, 
                            plot_any=False, 
                            batch_size=512, 
                            nr_of_batches=100,
                            save_to_file=True,
                            run_on_training=False,
                            save_folder='',
                            show_no_words = False):

        assert (len(save_folder) == 0) or (save_to_file)
        
        X_tr = self.X_tr
        y_tr = self.y_tr


        if run_on_training:
            X_te = self.X_tr
            y_te = self.y_tr
            X_sensi_te = self.X_tr_sensitiveAtarget
        else:
            X_te = self.X_te
            y_te = self.y_te
            X_sensi_te = self.X_te_sensitiveAtarget

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
            if show_no_words == False:
                print('START',start, 'END', end)
            results_one = [run_per_test_case(i, X_tr, y_tr, X_te, y_te, X_sensi_te, sensitive_name, privilige_variable_value, \
                unprivilige_variable_value, pri_num_feature_fetched, un_pri_num_feature_fetched, self.feature2columnmap, \
                verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance, self.trees) for i in np.arange(start,end)]
            results.extend(results_one)

        l = len(results)
        ser_p = [pd.Series(results[i]['p_list'], name=results[i]['index']) for i in range(l)]
        df_p = pd.concat(ser_p,axis=1).transpose()
        df_p = (1-df_p) #correcting because somehow the p's are inversed

        ser_qa = [pd.Series(results[i]['qa'], name=results[i]['index']) for i in range(l)]
        df_qa = pd.concat(ser_qa,axis=1).transpose()

        ser_y = [pd.Series(results[i]['y_te'], name=results[i]['index']) for i in range(l)]
        df_y = pd.concat(ser_y,axis=1).transpose()
        
        ser_mc = [pd.Series(results[i]['max_conf'], name=results[i]['index']) for i in range(l)]
        df_mc = pd.concat(ser_mc,axis=1).transpose()

        df_X = pd.concat([results[i]['X_te'] for i in range(l)],axis=1).transpose()

        if save_to_file:
            df_p.to_csv('{}/{}_dataframe_p_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_qa.to_csv('{}/{}_dataframe_qa_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_y.to_csv('{}/{}_dataframe_y_test_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_mc.to_csv('{}/{}_dataframe_mc_{}.csv'.format(save_folder,new_feat_mode,ii))
            df_X.to_csv('{}/{}_dataframe_X_test_{}.csv'.format(save_folder,new_feat_mode,ii))

        return df_p, df_qa, df_y, df_mc, df_X # What does this part mean?

def run_per_test_case(test_case_id, X_tr, y_tr, X_te, y_te, X_sensi_te, sensitive_name, privilige_variable_value, \
	unprivilige_variable_value, pri_num_feature_fetched, un_pri_num_feature_fetched,feature2columnmap, \
	verbose, new_feat_mode, clf, start_time2, all_features, features_by_importance, forestProcess):
    # start_time_00 = time.time()
    if verbose >= 1:
        if test_case_id % 1 == 0:
            print('Test case', test_case_id,"of",len(y_te))
            print('Time passed', time.time()-start_time2)
            print('Mode', new_feat_mode)
            print()
    if X_sensi_te.iloc[test_case_id,feature2columnmap[sensitive_name]] == privilige_variable_value:
        # privilege group
        budget = pri_num_feature_fetched
    else:
        budget = un_pri_num_feature_fetched


    features = deepcopy(all_features)
    test_example_full = X_te.iloc[test_case_id, :].values.astype(float)
    test_example = test_example_full * np.nan # Initialized as all nan
    # time_stop1 = time.time()
    max_conf = clf.predict_proba(test_example_full.reshape(1,-1))[0][0]
    p = []
    question_asked = []
    if new_feat_mode == 'feat-imp':
        upper_bound = min(budget, len(features))
    else:
        upper_bound = len(features)
    # time_stop2 = time.time()

    previous_answers = [tree.total_leaf_id.copy() for tree in forestProcess]
    p_cur = -1
    for question_i in range(upper_bound):

        if new_feat_mode == 'random':
            new_feature = random.sample(features,1)[0]
            features.remove(new_feature)
        elif new_feat_mode == 'feat-imp':
            new_feature = features_by_importance[question_i]
        elif new_feat_mode == 'ask-town':
            new_feature = getTheNextBestFeature(forestProcess, features, test_example, previous_answers, p_cur, absolutes_on=False)
            features.remove(new_feature)
        elif new_feat_mode == 'abs-agg':
            new_feature = getTheNextBestFeature(forestProcess, features, test_example, previous_answers, p_cur)
            features.remove(new_feature)
        else:
            raise Exception('mode has not been implemented')
        p_dict, p_cur = calcPValuesPerTree(test_example_full, forestProcess, previous_answers, new_feature)

        p.append(p_cur)
        
        question_asked.append(new_feature)
        # test_example[new_feature] = test_example_full[new_feature]

        if verbose >= 3:
            print()  
            print('Test case', test_case_id,"of",len(y_tr), 'index', X_tr.index[test_case_id])
            print('Time passed', time.time()-start_time2)
            print("Nr of questions asked : ", question_i)
            print("P before classification : ", p[-1])
            print("feature asked : ", list(X_tr)[new_feature])
            print("feature number asked : ", new_feature)
            print("max conf", max_conf)
    # time_stop3 = time.time()

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




def ClassifyWithPartialFeatures(sampleData,tree, previous_answers, new_feature, only_norm_prob = False):
    # only_norm_prob is for accelerating
    value = sampleData[new_feature]
    l = len(tree.feature2threshold_list[new_feature])
    if l > 0:
        if value > tree.feature2threshold_list[new_feature][l - 1]:
            previous_answers -= tree.featureAndthreshold2delete_set[new_feature][np.inf]
        else:
            for i in range(l):
                if value <= tree.feature2threshold_list[new_feature][i]:
                    previous_answers -= tree.featureAndthreshold2delete_set[new_feature][tree.feature2threshold_list[new_feature][i]]
                    break
    # time1 = time.time()

    total_w = 0
    norm_w = 1
    total_probs = np.zeros(tree.tree_single_value_shape, dtype = np.float32)
    
    if only_norm_prob == False:
        for node in previous_answers:
            total_probs += tree.values[node]
            total_w += tree.weighted_samples[node]

        norm_w = total_w / tree.weighted_samples[tree.feature == -2].sum()
        norm_probs = total_probs[0]/total_probs[0].sum()
    else:
        for node in previous_answers:
            total_probs += tree.values[node]
        norm_probs = total_probs[0]/total_probs[0].sum()
    # time2 = time.time()
    # print("time 4", time1 - start_time)
    # print("time 5", time2 - time1)
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


    # p_list = [ClassifyWithPartialFeatures(test_example,tree, previous_answers[i], new_feature, only_norm_prob = True)[0] for i, tree in enumerate(forest)]

    p_list = []
    # sampleData,tree, previous_answers, new_feature, only_norm_prob = False
    for i, tree in enumerate(forest):
        # only_norm_prob is for accelerating
        value = test_example[new_feature]
        # assert np.isnan(value) == False, "Value error in ClassifyWithPartialFeatures"
        l = len(tree.feature2threshold_list[new_feature])
        if l > 0:
            if value > tree.feature2threshold_list[new_feature][l - 1]:
                previous_answers[i] -= tree.featureAndthreshold2delete_set[new_feature][np.inf]
            else:
                for j in range(l):
                    if value <= tree.feature2threshold_list[new_feature][j]:
                        previous_answers[i] -= tree.featureAndthreshold2delete_set[new_feature][tree.feature2threshold_list[new_feature][j]]
                        break
        # time1 = time.time()

        total_probs = np.zeros(tree.tree_single_value_shape, dtype = np.float32)
        list_previous = list(previous_answers[i])
        total_probs = np.sum(tree.values[list_previous], axis = 0)
        # for node in previous_answers[i]:
        #     total_probs += tree.values[node]
        norm_probs = total_probs[0]/total_probs[0].sum()
        # time2 = time.time()
        # print("time 4", time1 - start_time)
        # print("time 5", time2 - time1)
        p_list.append(norm_probs[0])

    return p_list, np.mean(p_list)

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
