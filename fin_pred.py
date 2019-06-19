# -*- coding: utf-8 -*-
"""
@author: Chunghan
"""
################################ Load Dependencies ################################
from fin_pred_prelim import split_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import xgboost
import pandas as pd
import os
import ast
import string

assert 'fwb.csv' in os.listdir(), "Please set working directory to the financial-health folder so we can access relevant data (fwb.csv)"
    
data = pd.read_csv('fwb.csv').drop(['PUF_ID', 'sample','fpl', 'finalwt'],axis=1)
data=data[~data['FWBscore'].str.contains('R')]
random_seed=1000

model_specs = pd.read_csv('./greedy_10_size_5_scores/model_specs.csv')



################################ Define Variables and Functions ################################
#Get full list of predictors used. Know what each column is asking. Get all possible responses.
full_predictor_dict = {'ABSORBSHOCK':'Confidence in ability to raise $2,000 in 30 days',
     'ACT1_2':'I follow-through on financial goals I set for myself',
     'BENEFITS_2':'401(k) or Other Employer-Sponsored Retirement Savings Account',
     'EMPLOY1_2':'Work full-time for an employer or the military',
     'ENDSMEET':'#Difficulty of covering monthly expenses and bills',
     'FS1_1':'I know how to get myself to follow through on my financial intentions',
     'FS1_6':'I know how to keep myself from spending too much',
     'FWB1_1':'I could handle a major unexpected expense',
     'FWB1_3':'Because of my money situation...I will never have the things I want in life',
     'FWB1_4':'I can enjoy life because of the way I’m managing my money',
     'FWB1_5':'I am just getting by financially',
     'FWB1_6':'I am concerned that the money I have or will save won’t last',
     'FWB2_1':'Giving a gift...would put a strain on my finances for the month',
     'FWB2_2':'I have money left over at the end of the month',
     'FWB2_3':'I am behind with my finances',
     'GOALCONF':'Confidence in own ability to achieve financial goals',
     'HHEDUC':'Highest level of education of all household members',
     'MANAGE1_1':'Paid all your bills on time',
     'MANAGE1_3':'Paid off credit card balance in full each month',
     'MANAGE1_4':'Checked your statements, bills and receipts to make sure there were no errors',
     'MATHARDSHIP_6':'Utilities shut off due to non-payment',
     'PRODUSE_2':'Used pawn loan or auto title loan',
     'PROPPLAN_1':'I consult my budget to see how much money I have left',
     'RETIRE':'Actual date of retirement vs date planned',
     'SAVEHABIT':'Putting money into savings is a habit for me',
     'SAVINGSRANGES':'How much money do you have in savings today...?',
     'SNAP':'Any household member received SNAP benefits',
     'SOCSEC2':'At what age did you begin receiving benefits?',
     'SOCSEC3':'Age likely to start receiving Social Security retirement benefits',
     'SUBKNOWL1':'How would you assess your overall financial knowledge?',
     'VALUERANGES': 'If you were to sell your home today, what do you think it would be worth?',
     'agecat':'Age'}

response_description = {'FWBscore' : 'financial_well_being_score',
     'PPINCIMP':'household_income',
     'SCFHORIZON':'financial_planning_time_horizon',
     'MANAGE1_1':'paid_all_your_bills_on_time',
     'MANAGE1_2':'spend_within_budget_or_plan',
     'MANAGE1_3':'pay_off_credit_card_each_month',
     'ACT1_2':'follow_through_on_financial_goals',
     'EMPLOY':'employment_status'
    }

possible_choices={k:list(data.loc[:,k].unique()) for k in full_predictor_dict.keys()}

def get_best_params(summary,response_name):
    """
    Retrieve the specifications of the models that are achieve the highest cross validation score
    Input:
        summary (Pandas Dataframe): Input data frame that contains the cross validation score of models with different specifications 
        response_name (String): The name of the column that will be the response variable
    Output: Tuple of three elements:
        1. parameters: Dictionary of model parameters and corresponding values
        2. predictor list: List of predictor names that are best used for prediction
        3. model type: String that indicates the type of model being used to fit the data
    """
    temp=summary[summary['target'].str.contains(response_name)]
    temp_highest=temp.loc[temp['score'].idxmax()]
    params = ast.literal_eval(temp_highest['params'])
    predictors = temp_highest['predictors'].split('++')
    return (params, predictors, temp_highest['model'])


def final_fit(x, y, params, model_type):
    """
    Fit model using final specifications
    Input:
        x (Pandas Dataframe): Input set of predictors to use. The categorical predictors MUST be encoded into dummy variables beforehand
        y (Array): A vector of response variables, either in continuous or categorical format.
        params (Dictionary): Dictionary of parameter specifications
        model_type (String): String that indicates the type of model being used to fit the data
    Output: A trained model based on model_type
    """
    #Initiate model
    if model_type == 'rf':
        final_mod=RandomForestClassifier(max_depth= params['max_depth'], 
                                             n_estimators=params['n_estimators'],
                                             random_state=random_seed)
    elif model_type == 'glm':
        final_mod=LogisticRegression(C=params['C'],
                                     solver= params['solver'],
                                     multi_class='multinomial', 
                                     random_state=random_seed)
    elif model_type == 'xgb':
        final_mod=xgboost.XGBRegressor(
            booster=params['booster'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=random_seed)
    else:
        raise ValueError('model_type incorrectly specified')
    #Fit model
    final_mod.fit(x,y)
    return final_mod



def select_option(predictor):
    """
    Make options available for each question/survey item
    Input:
        predictor (String): The (abbreviated) name for the predictor or survey item
    Output: The option being selected, in the form of a string. It will be used for prediction later 
    """
    #Repeat until user selects a valid choice
    while True:
        #Explain the variable
        print('\n-------------------------------------')
        print('Please select a choice from below to answer the following survey question: "' + full_predictor_dict[predictor]+'"')
        #List the possible choices to select
        item_index=[string.ascii_lowercase[a] for a in range(len(possible_choices[predictor]))]
        item_content=sorted(possible_choices[predictor])
        choices= [a+'. '+b for a,b, in zip(item_index,item_content)]
        print('\n'.join(choices))
        selected_index = str(input("Please select a choice by typing in the corresponding alphabet: "))
        if selected_index in item_index:
            print('\n You selected the option.... ' + choices[item_index.index(selected_index)])
            break
        else:
            print('Please select a valid alphabet from '+item_index[0]+ ' to '+ item_index[-1])
            continue
    return item_content[item_index.index(selected_index)]



def make_prediction(data, selected_response, predictors,params, model_type, selection_dict,make_dummy=True):
    """
    Take user input and make prediction using the selected model
    Input:
        data (Pandas Dataframe): Input data that contains both the independent and response variables. Categorical predictors do not need to be encoded into dummy variables beforehand
        selected_response (String): The name of the column that will be the response variable
        predictor (Array): An array of strings containing the predictors that we will be interested in finding the conbinations of and searching over
        params (Dictionary): Dictionary of parameter specifications
        model_type (String): String that indicates the type of model being used to fit the data
        make_dummy (Boolean): Check if we need to convert some of the predictors into dummy variables first
    Output: A tuple of predicted result and prediction quality calculated from R^2 for Regressor and classification accuracy for Classifier
    """
    #Pre-processing step to build model for prediction
    x_train, x_test, y_train, y_test, encoder,encoding_order = split_data(data=data, 
                                              response_name=selected_response, 
                                              predictor_list=predictors, 
                                              model_type=model_type,
                                              make_dummy=make_dummy,
                                              random_seed=random_seed)
    final_model=final_fit(x_train, y_train,params, model_type)
    #Make prediction using the model
    input_data=np.array([selection_dict[p] for p in encoding_order]).reshape(1,-1)
    input_data=encoder.transform(input_data)
    pred=final_model.predict(input_data)
    #Use model quality as proxy to check how confident we are at making this prediction
    confidence=final_model.score(x_test,y_test) 
    return (pred,confidence)


def give_feedback(confidence_level, predicted_value, selected_response):
    """
    Based on the prediction results, print out feedback to users
    Input:
        confidence_level (Numeric): A number that indicates the model quality based on test data
        predicted_value (Array): An one-item array that contains the prediction result based on user unput
        selected_response (String): The name of the column that will be the response variable
    Output: None. This function prints the results
    """
    #Distinguish the printed output for different types of response variables
    if selected_response =='FWBscore':
        predicted_value = str(np.round(predicted_value[0]))+'/100' 
    else:
        predicted_value= str(predicted_value[0])
    #Print evaluation result
    print('\n***********************************************')
    if confidence_level <.3:
        confidence_string = 'low'
    elif confidence_level <.5:
        confidence_string = 'moderate'
    elif confidence_level < .7:
        confidence_string = 'reasonable'
    else:
        confidence_string = 'high'
    if selected_response =='FWBscore':
        message='Your predicted result for "{}" is: {}, and we have {} confidence about this prediction. Our prediction explains around {}% of the variation in the data.'.format(response_description[selected_response],
                                        str(predicted_value),
                                        confidence_string,
                                        str(int(np.round(confidence_level,2)*100)))
    else:
        message='Your predicted result for "{}" is: {}, and we have {} confidence about this prediction. We expect to be correct around {}% of the time.'.format(response_description[selected_response],
                                        str(predicted_value),
                                        confidence_string,
                                        str(int(np.round(confidence_level,2)*100)))
    print(message)
    print('***********************************************')
    

def input_and_predict(data, selected_response, model_specs, make_dummy=True):
    """
    Solicit user input and make prediction using the selected model
    Input:
        data (Pandas Dataframe): Input data that contains both the independent and response variables. Categorical predictors do not need to be encoded into dummy variables beforehand
        selected_response (String): The name of the column that will be the response variable
        make_dummy (Boolean): Check if we need to convert some of the predictors into dummy variables first
    Output: None. Prints the results
    """
    params, predictors, model_type = get_best_params(model_specs, selected_response)
        
    #Solicit user detailed data
    selection_dict={}
    for var in predictors:
        selected_option=select_option(var)
        selection_dict[var]=selected_option
    
    #Analyze data and predict outcome
    predicted_value, confidence_level = make_prediction(data=data, 
                    selected_response=selected_response, 
                    predictors=predictors,
                    params=params, 
                    model_type=model_type, 
                    make_dummy=make_dummy,
                    selection_dict=selection_dict)
    
    #Evaluation results
    give_feedback(confidence_level=confidence_level, 
                  predicted_value= predicted_value,
                  selected_response=selected_response)



def run_fin_pred(data, model_specs):
    """
    Run financial prediction feature. Ask for a response to predict, and them solicit relevant questions to make the prediction.
    Input:
        data (Pandas Dataframe): Input data that contains both the independent and response variables. Categorical predictors do not need to be encoded into dummy variables beforehand
        model_specs (Pandas Dataframe): Pre-determined set of model specification for each response vaiable that we want to predict
    Output: None. This function prints the instructions and results
    """
    while True:
        print('=========================================================')
        #Start by asking user what to predict. Input is not case sensitive
        selected_response = str(input("You can select one of the items below for us to predict \n" + '\n'.join([a + ' :\t' + b for a,b in response_description.items()]) + '\nPlease enter the abbreviated code to select, for example, EMPLOY for employment status. Enter code here: '))
        #Start predicting the selected response variable
        response_list=list(response_description.keys())
        if selected_response.lower() in [k.lower() for k in response_list]:
            selected_response = response_list[[k.lower() for k in response_list].index(selected_response.lower())]
            input_and_predict(data=data, 
                              selected_response=selected_response, 
                              model_specs=model_specs,
                              make_dummy=True)
        #Redirect if user inputs wrong name for the variable
        else:
            print("\n"+ selected_response +" is not a valid code. Please enter a code listed, such as FWBscore or MANAGE1_1")
            continue
        #Asking if the user wants to quit
        quit = str(input('Would you want to continue predicting or quit? Enter Y if you want to continue, type any other letter to end the program: '))
        if quit!='Y':        
            print('\nSee you next time!')
            break
        

