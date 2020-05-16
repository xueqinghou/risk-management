### -*- coding: utf-8 -*-
##"""
##Spyder Editor
##
##This is a temporary script file.
##"""
##
#import pandas as pd
#import numpy as np
#data = pd.read_csv('heloc.csv', header=0)
#data['RiskPerformance'][data['RiskPerformance']=='Good']=1
#data['RiskPerformance'][data['RiskPerformance']=='Bad']=0
#y = data['RiskPerformance'].astype('int')
#X = data.iloc[:,1:24]
#
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import StratifiedKFold 
#from sklearn.model_selection import cross_val_score
#np.random.seed(1)
#xTrain, xTest, yTrain, yTest = train_test_split(X,y, random_state = np.random.seed(1))
#
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
#
#from sklearn.base import BaseEstimator, TransformerMixin
class CategoricalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
     
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
     
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out
##        
#from sklearn.pipeline import Pipeline
#from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin): 
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
#
#cat_attributes = ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']
#num_attributes=['ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen',
#                         'AverageMInFile','NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec',
#                         'PercentTradesNeverDelq','MSinceMostRecentDelq','NumTotalTrades',
#                         'NumTradesOpeninLast12M','PercentInstallTrades','MSinceMostRecentInqexcl7days','NumInqLast6M','NumInqLast6Mexcl7days',
#                         'NetFractionRevolvingBurden','NetFractionInstallBurden','NumRevolvingTradesWBalance','NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization',
#                         'PercentTradesWBalance']
#num_pipeline = Pipeline([
#    ('selector',DataFrameSelector(num_attributes)),
#    ('std_scaler', StandardScaler()),])
#
#cat_pipeline = Pipeline([
#    ('selector',DataFrameSelector(cat_attributes)),
#    ('label_binarizer',CategoricalEncoder(encoding="onehot-dense")),])
#
#full_pipeline = FeatureUnion(transformer_list=[
#    ('num_pipeline',num_pipeline),
#    ('cat_pipeline',cat_pipeline),])
#
#data_train = full_pipeline.fit_transform(xTrain)
#
#data_test = full_pipeline.transform(xTest)
#
##data_train and data_test are both array
#
#from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC
#from sklearn.linear_model import LogisticRegression
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
#    
## The function `init_classifiers` returns a list of classifiers to be trained on the datasets
#def init_classifiers():
#    return([(SVC(), model_names[0], param_grid_svc), 
#            (LogisticRegression(), model_names[1], param_grid_logistic),
#            (KNeighborsClassifier(), model_names[2], param_grid_knn),
#            (GaussianNB(), model_names[3], param_grid_nb),
#            (DecisionTreeClassifier(), model_names[4], param_grid_tree),
#            (RandomForestClassifier(), model_names[6], param_grid_rf),
#            (AdaBoostClassifier(), model_names[7], param_grid_boost),
#            (LinearDiscriminantAnalysis(),model_names[8],param_grid_lda),
#            (MLPClassifier(),model_names[9],param_grid_nn)
#           ])
#
## 'model_names' contains the names  that we will use for the above classifiers
#model_names = ['SVM','LR','KNN','NB','Tree','QDA','RF','Boosting','LDA','NN']
#
## the training parameters of each model
#param_grid_svc = [{'C':[0.1,1],'kernel':['rbf','linear'], 'max_iter':[-1],'random_state':[1]}]
#param_grid_logistic = [{'C':[0.1,1], 'penalty':['l1','l2'],'random_state':[1]}]
#param_grid_knn = [{},{'n_neighbors':[1,2,3,4]}]
#param_grid_nb = [{}]
#param_grid_tree = [{'random_state':[1]},{'criterion':['gini'], 'max_depth':[2,3], 'min_samples_split':[3,5],'random_state':[1]}]
#param_grid_rf = [{'random_state':[1]},{'n_estimators':[10,30],'max_features':[0.2, 0.3], 'bootstrap':[True],'random_state':[1]}]
#param_grid_boost = [{'random_state':[1]},{'n_estimators':[10,20,30,40],'learning_rate':[0.1,1],'random_state':[1]}]
#param_grid_lda = [{}]
#param_grid_nn = [{'hidden_layer_sizes':[(5, 5),(10, 10),(50, 50)], 'alpha':[0.0001],  'random_state':[1]}]
#
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV
#from sklearn.svm import SVC
#import numpy as np
#from sklearn.metrics import accuracy_score
#from sklearn.model_selection import StratifiedKFold 
#from sklearn.model_selection import cross_val_score
#np.random.seed(1)
#def evaluate_model(xTrain, xTest, yTrain, yTest, model, model_name, params): 
#    grid_search = GridSearchCV(model, param_grid = params, cv=3)
#    grid_search.fit(xTrain,yTrain)
#    cvres = grid_search.cv_results_ 
#    
#    final_model = grid_search.best_estimator_
#    final_predictions = final_model.predict(xTest)
#    accuracy = accuracy_score(yTest, final_predictions)
#
#    Test_Score = accuracy
#    Classifier = model_name
#    CV_Score = grid_search.best_score_
#    dic = {'Classifier': Classifier, 'Test Score': Test_Score,'CV Score':CV_Score,'Best':final_model}
#    return dic
#
#df_model_comparison = pd.DataFrame(columns = ['Classifier','Test Score','CV Score','Best'])
#
#for i in init_classifiers():
#    dic = evaluate_model(data_train,data_test,yTrain, yTest, i[0], i[1], i[2])
#    row = pd.Series({'Classifier': dic['Classifier'],'Test Score':dic['Test Score'], 'CV Score':dic['CV Score'], 'Best':dic['Best']})
#    df_model_comparison = df_model_comparison.append(row,ignore_index=True)
#
#
#pipe_Boosting = Pipeline([('data_cleaning',full_pipeline),('Boosting', df_model_comparison['Best'].loc[6])])
#print('Accuracy: ', pipe_Boosting.score(xTest, yTest))
#
#import pickle
#import warnings
#pickle.dump(xTrain, open('X_train.sav', 'wb'))
#pickle.dump(pipe_Boosting, open('pipe_Boosting.sav', 'wb'))
#pickle.dump(xTest, open('X_test.sav', 'wb'))
#pickle.dump(yTest, open('y_test.sav', 'wb'))

import streamlit as st
import pickle
import numpy as np

# Load the pipeline and data
X_test = pickle.load(open('X_test.sav', 'rb'))
y_test = pickle.load(open('y_test.sav', 'rb'))

dic = {0: 'Bad', 1: 'Good'}

##Function to test certain index of dataset
def test_demo(index):
    values = X_test.iloc[index].astype(float)  # Input the value from dataset
    # Create four sliders in the sidebar
    a = st.sidebar.slider('ExternalRiskEstimate', -9.0, 100.0, values[0], 1.0)
    b = st.sidebar.slider('MSinceOldestTradeOpen', -9.0, 810.0, values[1], 1.0)
    c = st.sidebar.slider('MSinceMostRecentTradeOpen', -9.0, 400.0, values[2], 1.0)
    d = st.sidebar.slider('AverageMInFile', -9.0, 400.0, values[3], 1.0)
    e = st.sidebar.slider('NumSatisfactoryTrades', -9.0, 80.0, values[4], 1.0)
    f = st.sidebar.slider('NumTrades60Ever2DerogPubRec', -9.0, 20.0, values[5], 1.0)
    g = st.sidebar.slider('NumTrades90Ever2DerogPubRec', -9.0, 20.0, values[6], 1.0)
    h = st.sidebar.slider('PercentTradesNeverDelq', -9.0, 100.0, values[7], 1.0)
    i = st.sidebar.slider('MSinceMostRecentDelq', -9.0, 90.0, values[8], 1.0)
    j = st.sidebar.slider('MaxDelq2PublicRecLast12M', -9.0, 10.0, values[9], 1.0)
    k = st.sidebar.slider('MaxDelqEver', -9.0, 8.0, values[10], 1.0)
    l = st.sidebar.slider('NumTotalTrades', -9.0, 110.0, values[11], 1.0)
    m = st.sidebar.slider('NumTradesOpeninLast12M', -9.0, 20.0, values[12], 1.0)
    n = st.sidebar.slider('PercentInstallTrades', -9.0, 100.0, values[13], 1.0)
    o = st.sidebar.slider('MSinceMostRecentInqexcl7days', -9.0, 30.0, values[14], 1.0)
    p = st.sidebar.slider('NumInqLast6M', -9.0, 70.0, values[15], 1.0)
    q = st.sidebar.slider('NumInqLast6Mexcl7days', -9.0, 70.0, values[16], 1.0)
    r = st.sidebar.slider('NetFractionRevolvingBurden', -9.0, 240.0, values[17], 1.0)
    s = st.sidebar.slider('NetFractionInstallBurden', -9.0, 480.0, values[18], 1.0)
    t = st.sidebar.slider('NumRevolvingTradesWBalance', -9.0, 40.0, values[19], 1.0)
    u = st.sidebar.slider('NumInstallTradesWBalance', -9.0, 30.0, values[20], 1.0)
    v = st.sidebar.slider('NumBank2NatlTradesWHighUtilization', -9.0, 20.0, values[21], 1.0)
    w = st.sidebar.slider('PercentTradesWBalance', -9.0, 100.0, values[22], 1.0)

    #    #Print the prediction result
    pipe = pickle.load(open('pipe_Boosting.sav', 'rb'))
    aaa = pd.DataFrame(np.array([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w]).reshape(1, -1), columns =('ExternalRiskEstimate','MSinceOldestTradeOpen','MSinceMostRecentTradeOpen',
                         'AverageMInFile','NumSatisfactoryTrades','NumTrades60Ever2DerogPubRec','NumTrades90Ever2DerogPubRec',
                         'PercentTradesNeverDelq','MSinceMostRecentDelq','MaxDelq2PublicRecLast12M', 'MaxDelqEver','NumTotalTrades',
                         'NumTradesOpeninLast12M','PercentInstallTrades','MSinceMostRecentInqexcl7days','NumInqLast6M','NumInqLast6Mexcl7days',
                         'NetFractionRevolvingBurden','NetFractionInstallBurden','NumRevolvingTradesWBalance','NumInstallTradesWBalance','NumBank2NatlTradesWHighUtilization',
                         'PercentTradesWBalance'))
    res = pipe.predict(aaa)[0]
    st.write('Prediction:  ', dic[res])
    #    pred = pipe.predict(X_test)
    score = pipe.score(X_test, y_test)
    #    #cm = metrics.confusion_matrix(y_test, pred)
    st.write('Accuracy: ', score)
    
# if __name__ == "__main__":

#    #st.write('Confusion Matrix: ', cm)

# title
st.title('Credit Risk')

# explaination of features
#pd.set_option('display.max_columns',50)
#pd.set_option('display.max_rows',300)
#pd.set_option('display.width',1000) 
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#pd.set_option('max_colwidth',100)
#data_dic = pd.read_excel('heloc_data_dictionary.xlsx')
#data_dic = data_dic.iloc[0:24, 0:2]
#if st.checkbox('Show data dictionary'):
#    st.write(data_dic)
         
# show data
if st.checkbox('Show dataframe'):
    st.write(X_test)
# st.write(X_train) # Show the dataset

# explaination of special values
st.write('Explaination of special values:')
st.write('  -9 No Bureau Record or No Investigation')
st.write('  -8 No Usable/Valid Accounts Trades or Inquiries')         
st.write('  -7 Condition not Met (e.g. No Inquiries, No Delinquencies)')

number = st.text_input('Choose a row of information in the dataset:', 10)  # Input the index number

test_demo(int(number))  # Run the test function




