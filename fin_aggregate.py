# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:34:24 2019

@author: Chunghan
"""

################################ Load Dependencies ################################
from fin_pred import run_fin_pred, data, model_specs
from fin_health_check import userHealthCheck, runExample
import time
import sys


################################ Run Program ################################
def run_program():
    while True:    
        func = str(input("Hi! If you want to run financial check, please enter 'c'. If you want an estimation of your financial condition, please enter 'p': "))
        if func =='c':
            skip = str(input("Great! If you would like to run the check using an example, please press 'Y', otherwise enter any letter: "))
            if skip == 'Y':
                runExample()
            else:
                userHealthCheck()
            break
        elif func=='p':
            run_fin_pred(data,model_specs)
            break
        else:
            print("Sorry, you should enter a letter of either 'c' or 'p', please try again")
            sys.stdout.flush()
            time.sleep(2)
if __name__ == '__main__':
    run_program()