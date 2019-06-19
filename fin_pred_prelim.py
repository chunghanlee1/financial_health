# -*- coding: utf-8 -*-
"""
@author: Chunghan
"""

################################ Load Dependencies ################################
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from itertools import combinations
import numpy as np
import xgboost
import pandas as pd
import os

################################ Define Functions ################################
def tune_model(x_matrix, response, model_type,predictor_list = [], grid_params={}, cv=10):
    """
    Create a grid search model given data input (i.e. predictor_matrix and response)
    Input:
        predictor_matrix (Pandas Dataframe): A matrix of predictors, with the categorical variables coded into dummy variables
        response (Array): A vector of response variables, either in continuous or categorical format
        predictor_list (List): An array of strings specifying the columns of the predictor matrix that will be used in the model
        grid_params (Dictionary): A dctionary with parameters to grid search over. Keys are parameter name, values are arrays of specific parameter values
        model_type (String): The type of model to fit. Use rf and glm for categorical data (rf is random forest, glm is multinomial logit). Use xgb for continuous data (xgb is boosted regression).
        cv (Integer): Number of folds of cross validation for grid search
    Output: A GridSearchCV object that contains information on preferred parameters
    """
    x= x_matrix.loc[:,x_matrix.columns.isin(predictor_list)]
    if model_type == 'rf':
        mod=RandomForestClassifier(random_state=random_seed)
    elif model_type == 'glm':
        mod=LogisticRegression(multi_class='multinomial', random_state=random_seed)
    elif model_type == 'xgb':
        mod= xgboost.XGBRegressor(random_state=random_seed)
    grid = GridSearchCV(mod, grid_params, cv=cv)
    grid.fit(x ,response)
    return grid


def enumerate_predictors(x, y, min_size, max_size, model_type, grid_params, top_mod=10,cv=10, greedy_threshold=5, selected_predictors=[]):
    """
    Enumerate all combinations of predictors to find the best predicting 
    variables. The range of acceptable model sizes that we will consider
    will be controlled by min_size and max_size. Since combinations can 
    be unpractically large, a greedy approach is used where only the predictors
    listed in selected_predictors are considered. When the selected_predictors
    is not specified, it will contain the predictors that individually contribute to the 
    most explanatory power.
    Input:
        x (Pandas Dataframe): Input set of predictors to use. The categorical predictors do not need to be encoded into dummy variables beforehand
        y (Array): A vector of response variables, either in continuous or categorical format.
        min_size (Integer): The smallest model size or number of predictors to consider
        max_size (Integer): The largest model size or number of predictors to consider
        grid_params (Dictionary): A dctionary with parameters to grid search over. Keys are parameter name, values are arrays of specific parameter values
        model_type (String): The type of model to fit. Use rf and glm for categorical data (rf is random forest, glm is multinomial logit). Use xgb for continuous data (xgb is boosted regression).
        cv (Integer): Number of folds of cross validation for grid search
        top_mod (Integer): The number of best performing models that we want to track
        greedy_threshold (Integer): The number of items in selected_predictors
        selected_predictors (Array): An array of strings containing the predictors that we will be interested in finding the conbinations of and searching over
    Output: List of list that contains information on models with the highest cross validation scores (the score itself, the model, and predictors used)
    """
    assert max_size<=greedy_threshold
    
    best_mod_list = []
    
    for size in range(min_size,max_size+1):
        if size == 1:
            to_loop = x.columns
            tot_elements = len(x.columns)
        elif size>1:
            if selected_predictors:
                #selected_predictors is sorted list of list that contains scores and correponding individual predictors
                selected_predictors.sort(key= lambda x: x[0], reverse=True)
                selected_predictors=selected_predictors[:greedy_threshold]
                to_loop = combinations([item[1] for item in selected_predictors], size)
                tot_elements = sum(1 for _ in combinations([item[1] for item in selected_predictors], size))
            else:
                to_loop = combinations(x.columns, size)
                tot_elements = sum(1 for _ in combinations(x.columns, size))
            
        
        for i, predictor in enumerate(to_loop):
            
            print('Have ' + str(tot_elements - i ) + ' sets remaining for model size ' + str(size))
                        
            x_dummy=pd.get_dummies(x.loc[:,predictor])
            
            mod = tune_model(predictor_matrix = x_dummy,
                       model_type=model_type,
                        response= y, 
                        predictor_list= x_dummy.columns,
                        grid_params=grid_params,
                        cv=cv)
            
            if len(best_mod_list) < top_mod:
                best_mod_list.append([mod.best_score_, mod, predictor])
            else:
                if mod.best_score_ > best_mod_list[-1][0]:
                    best_mod_list.pop()
                    best_mod_list.append([mod.best_score_,mod, predictor])
            
            if size == 1:
                if len(selected_predictors) < greedy_threshold:
                    selected_predictors.append([mod.best_score_, predictor])
                else:
                    if mod.best_score_ > selected_predictors[-1][0]:
                        selected_predictors.pop()
                        selected_predictors.append([mod.best_score_, predictor])
            
            best_mod_list.sort(key= lambda x: x[0], reverse=True)
            selected_predictors.sort(key= lambda x: x[0], reverse=True)
            
    return best_mod_list



def preprocess_data(data, response_name):
    """
    Preprocess data for categorical response variables that are one of the columns in data
    Input:
        data (Pandas Dataframe): Data that contains all predictors used and a column for response variable
        response_name (String): Name of the column that contains the response variable
    Output: A tuple with the first element being a Pandas Dataframe of predictors and second element an Array of categorical variables for the response
    """
    y = data[response_name].astype('category')
    x_ind = data.columns != response_name
    x = data.loc[:, x_ind]
    return (x,y)


def predict_non_FWB(data, response_name,min_size, max_size, model_type, grid_params, top_mod=10,cv=10, greedy_threshold=5, selected_predictors=[]):
    """
    Wrapper function that extracts the response variables from the data and then perform grid search.
    Input:
        data (Pandas Dataframe): Input data that contains both the independent and response  variables. Categorical predictors do not need to be encoded into dummy variables beforehand
        response_name (String): The name of the column that will be the response variable
        min_size (Integer): The smallest model size or number of predictors to consider
        max_size (Integer): The largest model size or number of predictors to consider
        grid_params (Dictionary): A dctionary with parameters to grid search over. Keys are parameter name, values are arrays of specific parameter values
        model_type (String): The type of model to fit. Use rf and glm for categorical data (rf is random forest, glm is multinomial logit). Use xgb for continuous data (xgb is boosted regression).
        cv (Integer): Number of folds of cross validation for grid search
        top_mod (Integer): The number of best performing models that we want to track
        greedy_threshold (Integer): The number of items in selected_predictors
        selected_predictors (Array): An array of strings containing the predictors that we will be interested in finding the conbinations of and searching over

    Output: List of list that contains information on models with the highest cross validation scores (the score itself, the model, and predictors used)
    """
    x, y =preprocess_data(data, response_name)
    mod_list = enumerate_predictors(x = x,
                         y = y,
                         min_size = min_size,
                         max_size=max_size,
                         model_type=model_type,
                         grid_params=grid_params,
                         top_mod=top_mod,
                         cv=cv,
                         greedy_threshold=greedy_threshold,
                         selected_predictors=selected_predictors)
    return mod_list



def split_data(data,response_name, predictor_list, model_type, make_dummy, random_seed=2000):
    """
    Extract needed data and then split into training and testing
    Input:
        data (Pandas Dataframe): Input data that contains both the independent and response variables. Categorical predictors do not need to be encoded into dummy variables beforehand
        response_name (String): The name of the column that will be the response variable
        predictor_list (Array): An array of strings containing the predictors that we will be interested in finding the conbinations of and searching over
        model_type (String): String that indicates the type of model being used to fit the data
        make_dummy (Boolean): Check if we need to convert some of the predictors into dummy variables first
    Output: Tuple of at least four elements, predictors and response separated in train and test portions
            If make_dummy is True, then return two additional dummy variable encoding specifications
    """
    if model_type == 'rf':
        temp_data = data.loc[:, data.columns.isin([response_name]+predictor_list)]
        x_data,y_data=preprocess_data(temp_data, response_name)
    
    elif model_type == 'glm':
        temp_data = data.loc[:, data.columns.isin([response_name]+predictor_list)]
        x_data,y_data=preprocess_data(temp_data, response_name)
    
    elif model_type == 'xgb':
        y_data = data.loc[:,'FWBscore'].astype(int)
        x_data = data.loc[:, predictor_list]
    if make_dummy:
        encoder=OneHotEncoder(categories='auto',
                      handle_unknown= 'ignore',
                      sparse=False)
        encoding_order = x_data.columns
        x_data=encoder.fit_transform(x_data)
    x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=.01, random_state=random_seed)
    if make_dummy:
        return (x_train, x_test, y_train, y_test, encoder, encoding_order)
    else:
        return (x_train, x_test, y_train, y_test)



################################ Build Model ################################
#The code below conducts preliminery analysis to find suitable models to fit the data
if __name__ == '__main__':
    
    response_variable_list = ['FwbScore', #Financial Well-being score
                          'PPINCIMP', #Household income
                          'SCFHORIZON', #Financial planning time horizon
                          'MANAGE1_1', #Paid all your bills on time
                          'MANAGE1_2', #Stayed within your budget or spending plan
                          'MANAGE1_3', #Paid off credit card balance in full each month
                          'ACT1_2', #I follow-through on financial goals I set for myself
                          'EMPLOY' #Primary or only employment status
                          ]

    assert 'fwb.csv' in os.listdir(), "Please set working directory to the financial-health folder so we can access relevant data"
    
    data = pd.read_csv('fwb.csv').drop(['PUF_ID', 'sample','fpl', 'finalwt'],axis=1)
    data=data[~data['FWBscore'].str.contains('R')]
    random_seed=1000

    x_train, x_test, y_train, y_test = split_data(data=data, 
                                                  response_name='FWBscore', 
                                                  predictor_list=data.columns[data.columns != 'FWBScore'], 
                                                  model_type='xgb',
                                                  make_dummy=False)

    #Predict Financial Well-being--------------
    #Use XGBoost Regressor
    min_size=1
    max_size=5
    xgb_parameters={'max_depth':range(4,11,2), 
                 'booster':['gbtree','gblinear'], 
                 'learning_rate':np.arange(0,1,0.2)}
    #Select predictors for well being prediction
    xgb_list=enumerate_predictors(x = x_train,
                         y = y_train,
                         min_size=min_size,
                         max_size=max_size,
                         model_type='xgb',
                         grid_params=xgb_parameters,
                         top_mod=10,
                         cv=5,
                         greedy_threshold=10)
    total_df = [['xgb','FwbScore',a,b.best_params_, '++'.join(list(c))] for a,b,c in xgb_list]
    total_df = pd.DataFrame(total_df).rename(columns={0:'model',1:'target',2:'score',3:'params',4:'predictors'})
    
    
    
    #Predict Other Financial Information-------------
    #Use RandomForest or Softmax Regression
    rf_parameters={'n_estimators':range(10,31,3) ,'max_depth':range(4,10,2)}
    glm_parameters={'solver':['sag'],'C':np.arange(0.1,1,0.1)}
    
    
    #Create full lists for 1 predictor only for future greedy search for each of glm, rf, and xgboost
    #Two models, all predictors, indexed by predictors. col_names predictor, rf and glm
    modelTypes=['rf','glm']
    
    #Search through models of different sizes using greedy approach
    min_size=1
    max_size=5 #Choose 5 because beyond that the coefficients start to fail to converge
    greedy_threshold=10
    mod_list = []
    for t in response_variable_list[1:]:
        for m in modelTypes:
            
            print('Currently at predictor ' + t + ' and model '+ m)
            
            if m == 'rf':
                grid_params = rf_parameters
            elif m =='glm':
                grid_params = glm_parameters
                
            temp=predict_non_FWB(data=x_train,
                            response_name= t,
                            min_size = min_size,
                             max_size= max_size,
                             model_type=m,
                             grid_params=grid_params,
                             top_mod=10,
                             cv=5,
                             greedy_threshold=greedy_threshold)
            temp = [item+[t,m] for item in temp]
            mod_list.extend(temp)
    total_df2 = [[e,d,a,b.best_params_, '++'.join(list(c))] for a,b,c,d,e in mod_list]
    total_df2 = pd.DataFrame(total_df).rename(columns={0:'model',1:'target',2:'score',3:'params',4:'predictors'})
    total_df2 = pd.concat([total_df, total_df2], axis=0)
    #total_df2.to_csv( './greedy_10_size_5_scores/model_specs.csv', index=False, header=True)

