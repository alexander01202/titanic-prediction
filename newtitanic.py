import pandas as pd
import numpy as np
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

# file = open('train.csv')
#     df = pd.read_csv(file)

class General():
    sex = { 'male':1, 'female':2 }
    embarked = { "s":0 , "c":1, "q":2 }

    def __init__(self, df):
        self.df = df
        self.dict = {}
        self.symbols_dict = {}
    
    def gender(self, x):
        if x != '':
            x = x.lower()
            return General.sex[x]
        else:
            return x
    
    def embark(self, x):
        if x != '':
            x = x.lower()
            return General.embarked[x]
        else:
            return x

    def convert_letter_to_num(self,x):
        """
        Takes as input each value of a column and Converts Letters/Symbols to numbers
        This func has already been implemented in convert_to_num. CALL convert_to_num() instead
        """
        x = x.lower()
        x = x.replace(' ', '')
        is_symbol = any(symbol in x for symbol in self.symbols_dict)

        for symbol, num in self.symbols_dict.items():
            x = x.replace(symbol, f'{num}')
        for old, new in self.dict.items():
            x = x.replace(old, f'{new}')
        return x
    
    def fill_up_missing_rows(self):
        """
        Changing all the missing data from nan to an empty string
        This is done after we've stored/collected which columns had a lot of empty values
        """
        self.df.fillna('',inplace=True)

    def convert_to_num(self):
        """
        Takes nothing as input. Converts non-numeric val in Cabin,Ticket,Embarked,Sex to numbers.
        The numbers are in string format. Convert to int/float using convert_to_int/convert_to_float.
        Returns the entire dataframe
        """
        num = 10

        # Make a dictionary of all letter with values as their index + 1. 
        # This would help us convert letters to numbers
        for index,letter in enumerate(string.ascii_lowercase):
            self.dict[letter] = index + 1
        
        # Make a dictionary of all symbols with values starting from 10 upwards. 
        # This would help us convert symbols to numbers
        for char in (string.punctuation):
            self.symbols_dict[char] = int(num)
            num += 1

        self.df['Cabin'] = list(map(lambda x: self.convert_letter_to_num(x), self.df['Cabin']))
        # print(self.df['Cabin'],'self.df[Cabin]')
        self.df['Ticket'] = self.df['Ticket'].map(lambda x: self.convert_letter_to_num(x))
        # print(self.df['Ticket'],'self.df[Ticket]')
        self.df['Embarked'] = self.df['Embarked'].map(lambda x: self.embark(x))
        self.df['Sex'] = self.df['Sex'].map(lambda x: self.gender(x))
        
        # header = self.df.columns.values.tolist()
        # sex_index = self.df.columns.get_loc('Sex')
        # embarked_index = self.df.columns.get_loc('Embarked')
        # cabin_index = self.df.columns.get_loc('Cabin')
        # self.df.apply()
        # ticket_index = self.df.columns.get_loc('Ticket')
        # passenger_index = self.df.columns.get_loc('PassengerId')
        # name_index = self.df.columns.get_loc('Name')
        
        return self.df
    
    def input_mean_for_missing_rows(self):
        """
        Fills in an empty field with the mean of it's column.
        Changes the dataframe to the non-empty dataframe
        Returns the new dataframe
        """
        header = list(self.df.columns.values)
        for head in header:
            arr = [val for val in self.df[head] if val != '']
            self.df[head].mask(self.df[head] == '',inplace=True, other=mean(arr))
        
        return self.df
    
    

    def convert_to_int(self, columns):
        """
        Takes as input a dict of all columns to be converted to int type
        Returns nothing
        """
        for col in columns:
            col_val = self.df[col].tolist()
            for index,val in enumerate(col_val):
                if val != '':
                    col_val[index] = int(val)
            self.df[col] = col_val

    def convert_to_float(self, columns):
        """
        Takes as input a dict of all columns to be converted to float type
        Returns nothing
        """
        for col in columns:
            col_val = self.df[col].tolist()
            for index,val in enumerate(col_val):
                if val != '':
                    col_val[index] = float(val)
            self.df[col] = col_val

    def drop_col(self,col):
        """
        Drops any column of choice
        """
        self.df.drop([col],inplace=True, axis=1)

    def filter_columns(self,x,drop_col=False):
        """
        Returns column name if 30% or more of the column Values are missing
        """
        percent_of_df = len(self.df) * 0.3
        missing_rows = self.df[x].isnull().sum()
        if missing_rows > percent_of_df:
            if drop_col:
                self.df.drop([x], inplace=True, axis=1)
            return x
    
    def add_columns(self, **kwargs):
        """
        Takes as input arrays of dataframe columns needed to make inferences
        Returns entire dataframe with updated values
        """
        new_col = 0
        col_name = kwargs['col_name']
        col_1 = np.array(kwargs['col_1'])
        col_2 = np.array(kwargs['col_2'])
        

        if col_name == 'Fare_per_person':
            # print(kwargs['col_1'], '1Kwargs')
            # print(kwargs['col_2'], '2Kwargs')
            new_col = (col_1 + 1) / (col_2 + 1)
        elif col_name == 'Person_survival':
            new_col = ((col_1 / (col_2 + 1)) * (np.array(kwargs['col_3']) + 1)) * np.array(kwargs['col_4'])
        elif col_name == 'FamilySize':
            new_col = col_1 + col_2
        
        self.df[col_name] = new_col
        return self.df

def Train():
    TEST_SIZE = 0.3
    to_int = {'Pclass', 'Age', 'SibSp', 'Parch','Ticket','Cabin'}
    to_float = {'Fare'}

    # We join both to one dictionary because we need to convert the int columns to float first to get rid of the string
    both_to_float = to_int | to_float

    file = open('train.csv')
    df = pd.read_csv(file)
   
    gen = General(df)
    gen.drop_col('Name')

    # Getting all of the columns with 30% more nan/empty values
    remove_col = list(map(lambda x: gen.filter_columns(x),df.columns.values.tolist()))
    remove_col = [col for col in remove_col if col != None]
    
    # Replace any nan/empty value with an empty string
    gen.fill_up_missing_rows()
    
    # Converts Cabin,Ticket,Embarked,Sex Columns values to numbers
    gen.convert_to_num()

    # Float all numerical columns specified
    # This is to prepare for mean calculations
    gen.convert_to_float(both_to_float)
    
    # Convert specified col to int.
    # This is to prepare for mean calculations
    gen.convert_to_int(to_int)
    
    # Fill in fields with empty strings/values with the mean of the entire column
    gen.input_mean_for_missing_rows()
    
    for col in remove_col:
        gen.drop_col(col)
    
    gen.add_columns(col_1=gen.df['Parch'].tolist(),col_2=gen.df['SibSp'].tolist(), col_name='FamilySize')
    gen.add_columns(col_1=gen.df['FamilySize'].tolist(),col_2=gen.df['Fare'].tolist(), col_name='Fare_per_person')
    new_df = gen.add_columns(col_1=gen.df['Age'].tolist(),col_2=gen.df['SibSp'].tolist(), col_3=gen.df['Parch'].tolist(),col_4=gen.df['Sex'].tolist(), col_name='Person_survival')
    labels = gen.df['Survived'].values.tolist()
    evidence = new_df[new_df.columns.values.tolist()[2:]].values.tolist()
    # print(evidence, 'evidence')

    # print(gen.df.isnull().sum(),'df.isnull().sum()2')
    # print(evidence,'evidence')

    svc = svm.SVC(kernel='rbf', random_state=40)
    kf = KFold(n_splits=5, shuffle=True, random_state=43)
    transformer = normalize(evidence, axis=0)
    # print(transformer,'transformer')
    print(mean(cross_val_score(svc, evidence, labels, cv=kf)), 'Cross_val_score')
    X_train, X_test, y_train, y_test = train_test_split(
        transformer, labels, test_size=TEST_SIZE, random_state=42
    )
    
    # Train model and make predictions
    model = svc.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    # Plot Bar graph
    header = gen.df.columns.values.tolist()
    header = [head for head in header if head not in ['PassengerId','Name','Cabin']]
    perm_importance = permutation_importance(svc, X_test, y_test)
    feature_names = header
    features = np.array(feature_names)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    plt.show()
    print(accuracy_score(y_test, predictions),'accuracy_score(y_test, predictions)')

def Test(General):
    file = open('test.csv')
    df = pd.read_csv(file)

    # remove_col = df.columns.values.tolist().map(lambda x: Test.filter_columns(x))


Train()