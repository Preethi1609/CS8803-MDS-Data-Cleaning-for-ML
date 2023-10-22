from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data_runwalk(df):
   numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()[:-1]
   imputer = SimpleImputer(strategy='mean')
   df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
   return df

def preprocess_data_obesity(df):
   numeric_columns = ['Age', 'Height', 'Weight', 'FCVC', 'CH2O', 'FAF', 'TUE']
   categorical_columns = ['family_history_with_overweight', 'FAVC', 'NCP', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

   numeric_imputer = SimpleImputer(strategy='mean')
   df[numeric_columns] = numeric_imputer.fit_transform(df[numeric_columns])

   categorical_imputer = SimpleImputer(strategy='most_frequent')
   df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])


   columns_to_drop = ['SCC', 'MTRANS', 'SMOKE', 'CAEC'] 
   df = df.drop(columns=columns_to_drop)

   scaler = StandardScaler()
   df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

   return df
