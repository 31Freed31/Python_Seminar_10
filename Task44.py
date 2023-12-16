import random
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI':lst})

ohe = OneHotEncoder(sparse=False)
ohe_data = ohe.fit_transform(data[['whoAmI']])
ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out())
result = pd.concat([data, ohe_df], axis=1)

print(result.head())