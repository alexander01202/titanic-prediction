import os
import csv
import numpy as np
import math
import pandas as pd
import copy
from statistics import mean, mode
import string
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split,KFold, cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score

TEST_SIZE = 0.3
# imputer = KNNImputer(n_neighbors=10)
file = open('train.csv')
df = pd.read_csv(file)

def train():
    header = df.columns.values.tolist()
    header = [head for head in header if head not in ['PassengerId','Name','Cabin']]
    
    svc = svm.SVC(kernel='rbf', random_state=40)
    kf = KFold(n_splits=5, shuffle=True, random_state=43)
    evidence, labels = extract_features(df)
    transformer = normalize(evidence, axis=0)
    # print(transformer,'transformer')
    print(mean(cross_val_score(svc, evidence, labels, cv=kf)), 'Cross_val_score')
    X_train, X_test, y_train, y_test = train_test_split(
        transformer, labels, test_size=TEST_SIZE, random_state=42
    )
    
    # Train model and make predictions
    model = svc.fit(X_train, y_train)
    # model = train_model(X_train, y_train)
    
    predictions = model.predict(X_test)
    print(accuracy_score(y_test, predictions),'accuracy_score(y_test, predictions)')

    # Plot Bar graph
    perm_importance = permutation_importance(svc, X_test, y_test)
    feature_names = header
    features = np.array(feature_names)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()

    test_file = open('test.csv')
    new_df = pd.read_csv(test_file)

    evidence,PassengerId = extract_features(new_df)

    transformer = normalize(evidence, axis=0)
    new_predictions = model.predict(transformer)

    # pd.DataFrame({"PassengerId":PassengerId,"Survived":new_predictions}).to_csv('prediction.csv', index=False)

def extract_features(df):
    titanic_data = list()
    label_data = list()
    
    sex = { 'male':1, 'female':0 }
    embarked = { "s":0 , "c":1, "q":2 }
    my_dict = {}
    symbols_dict = {}
    num = 10
    for index,letter in enumerate("abcdefghijklmnopqrstuvwxyz"):
        my_dict[letter] = index + 1
    for char in (string.punctuation):
        symbols_dict[char] = int(num)
        num += 1

    convert_to_int = {'Pclass', 'Age', 'SibSp', 'Parch'}
    convert_to_float = {'Fare'}

    # impute = imputer.fit_transform(df)
    # print(impute,'impute')
    # df = df.fillna(value=df['Fare'].mean())
    # print(df['Cabin'].isnull().sum(),'df.isnull().sum()2')
    header = df.columns.values.tolist()
    # print(header, 'header')
    sex_index = header.index('Sex')
    embarked_index = header.index('Embarked')
    cabin_index = header.index('Cabin')
    ticket_index = header.index('Ticket')
    survive_index = [header.index(head) for head in header if head == 'Survived']
    passenger_index = header.index('PassengerId')
    name_index = header.index('Name')
    new_column = add_columns(df['Parch'].tolist(),df['SibSp'].tolist(), 'FamilySize')
    print(len(new_column.tolist()),'new_column')

    remove_rows = [passenger_index,name_index]
    if len(survive_index) > 0:
        convert_to_int.add('Survived')
        remove_rows.append(survive_index[0])
    percent_of_df = len(df) * 0.3

    for head in header:
        if header.index(head) in remove_rows:
            continue
        missing_rows = df[head].isnull().sum()
        if missing_rows > percent_of_df:
            remove_rows.append(header.index(head))
    # Changing all the missing data from nan to an empty string
    # This is done after the previous code because we need to know which rows were actually empty to begin with
    df = df.fillna('')

    df_all_int = list()
    for rows in df.values:
        cloned_row = copy.deepcopy(list(rows))
        # Get the row value by getting the index position from the header
        gender = cloned_row[sex_index].lower()
        cabin = rows[cabin_index].lower()
        embark = cloned_row[embarked_index].lower()
        ticket = cloned_row[ticket_index].lower()
        
        for old, new in my_dict.items():
            cabin = cabin.replace(old, f'{new}')
        for alpha, symbol in zip(my_dict.items(), symbols_dict.items()):
            ticket = ticket.replace(f'{alpha[0]}', f'{alpha[1]}')
            ticket = ticket.replace(f'{symbol[0]}', f'{symbol[1]}')
        ticket = ticket.replace(' ', '')
        cabin = cabin.replace(' ', '')
        
        cloned_row[sex_index] = int(sex[gender])
        if len(cloned_row[cabin_index]) > 0:          
            cloned_row[cabin_index] = int(cabin)

        if len(embark) > 0:
            cloned_row[embarked_index] = int(embarked[embark])

        if len(ticket) > 0:
            cloned_row[ticket_index] = int(ticket)
        df_all_int.append(cloned_row)
    
    # Seperate changing to integer from checking if the columns are integers/floats
    for rows in df_all_int:
        cloned_row = copy.deepcopy(list(rows))
        for index,row in enumerate(rows):
            if type(row) == str and len(row) < 1:
                values = [v_row for rows in df_all_int for i,v_row in enumerate(rows) if i == index and v_row != row]
                # print(mean(values),'values')
                # print(df[header[index]],'df[header[index]]')
                cloned_row[index] = round(mean(values))
        
        for int_row in convert_to_int:
            int_header_index = header.index(int_row)
            # print(int_row,'cloned_row[int_header_index]')
            cloned_row[int_header_index] = int(float(cloned_row[int_header_index]))
            
        for float_row in convert_to_float:
            float_header_index = header.index(float_row)
            cloned_row[float_header_index] = round(float(cloned_row[float_header_index]))
        
        # Testing set doesn't contain survived column only add survived colum in training set
        # Submission requires passenger number to be in the csv file
        if len(survive_index) > 0:
            # print(label_data)
            # print('survive_index')
            label_data.append(cloned_row[survive_index[0]])
        else:
            # print('Label')
            # print(label_data)
            label_data.append(cloned_row[passenger_index])
        new_cloned_row = [cloned_row[i] for i, e in enumerate(cloned_row) if i not in remove_rows]
        titanic_data.append(new_cloned_row)
    
    # df = pd.DataFrame(titanic_data)
    # mean = df.mean()
    # new_titanic_data = replace_empty_rows(mean,df,remove_rows)
    
    return (titanic_data, label_data)

def add_columns(col_1,col_2,col_name):
    col_1 = np.array(col_1)
    col_2 = np.array(col_2)
    df[col_name] = col_1 + col_2

    return df[col_name]

train()