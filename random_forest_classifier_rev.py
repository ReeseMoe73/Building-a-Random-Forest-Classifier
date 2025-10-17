#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load the library with the iris dataset
from sklearn.datasets import load_iris


# In[2]:


# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier


# In[3]:


# Load pandas
import pandas as pd


# In[4]:


# Load numpy
import numpy as np


# In[5]:


# Set random seed
np.random.seed(0)


# In[6]:


# Create an object called iris with the iris data
iris = load_iris()


# In[7]:


# Create a dataframe with the four feature variables
df=pd.DataFrame(iris.data,columns=iris.feature_names)


# In[8]:


# View the top 5 rows
df.head()


# In[9]:


# Add a new column with the species names; this is what we are going to try to predict
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)


# In[10]:


# View the top 5 rows
df.head()


# In[11]:


#Create a new column that, for each row, generates a random number between 0 and 1, and if that value is less than or equal to .75, then sets the value of that cell as True
#and false otherwise. This is a quick and dirty way of randomly assigning some rows to
#be used as the training data and some as the test data.

df['is_train'] = np.random.random(len(df)) < 0.75


# In[12]:


# View the top 5 rows
df.head()


# In[13]:


# Create two new dataframes, one with the training rows and one with the test rows
train,test=df[df['is_train']==True],df[df['is_train']==False]


# In[14]:


# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:',len(train))
print('Number of observations in the test data:',len(test))


# In[15]:


# Create a list of the feature column's names
features=df.columns[:4]


# In[16]:


# View features
features


# In[17]:


# train['species'] contains the actual species names. Before we can use it, # we need to convert each species name into a digit. So, in this case, there
# are three species, which have been coded as 0, 1, or 2.
y=pd.factorize(train['species'])[0]


# In[18]:


# View target
y


# In[19]:


# Create a random forest Classifier. By convention, clf means 'Classifier'
clf=RandomForestClassifier(n_jobs=2,random_state=0)


# In[20]:


# Train the Classifier to take the training features and learn how they relate to the training y (the species)
clf.fit(train[features],y)


# In[21]:


clf = RandomForestClassifier(
    bootstrap=True,
    class_weight=None,
    criterion="gini",
    max_depth=None,
    max_features="sqrt",          # instead of 'auto'
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,    # replacement for min_impurity_split
    min_samples_leaf=1,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=200,             # stronger default than 10
    n_jobs=2,
    oob_score=False,
    random_state=0,
    verbose=0,
    warm_start=False,
)


# In[22]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
import joblib

#Fallback: build train/test from Iris if not present
globals_dict = globals()
if not (("train" in globals_dict) and ("test" in globals_dict)):
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Normalize column names
    df.columns = [c.lower().replace(" (cm)", "").replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"target": "species"})
    target = "species"
    features = [c for c in df.columns if c != target]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    train = train_df.reset_index(drop=True)
    test  = test_df[features].reset_index(drop=True)  # test has features only
else:
    # If the user provided train/test attempt to infer/validate target/features next.
    pass

# Target and feature configuration 
if "target" not in globals_dict:
    # Try common label names
    candidates = ("target","label","class","species","y","outcome")
    found = [c for c in candidates if c in train.columns]
    if len(found) == 1:
        target = found[0]
    elif len(found) > 1:
        raise ValueError(f"Multiple plausible target columns found: {found}. Please set `target` explicitly.")
    else:
        raise ValueError(f"No typical target column found in train. Columns: {list(train.columns)}")

# If features not defined, default to all columns except target
if "features" not in globals_dict or not isinstance(globals_dict.get("features"), (list, tuple)):
    features = [c for c in train.columns if c != target]

# Validate presence/shape
missing_in_train = [c for c in features if c not in train.columns]
missing_in_test  = [c for c in features if c not in test.columns]
assert not missing_in_train, f"Features missing in train: {missing_in_train}"
assert not missing_in_test,  f"Features missing in test: {missing_in_test}"
assert target in train.columns, f"Target '{target}' not found in train.columns"

X = train[features].copy()
y = train[target].copy()

# Coerce object numerics 
for c in features:
    if X[c].dtype == "object":
        X[c]    = pd.to_numeric(X[c], errors="coerce")
        test[c] = pd.to_numeric(test[c], errors="coerce")

# Column typing 
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(X[c])]
cat_cols = [c for c in features if c not in num_cols]

#  Preprocessor + Model pipeline 
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

pipe = Pipeline([
    ("pre", pre),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

# Split, fit, validate
stratify_y = y if y.value_counts().min() >= 2 else None
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=stratify_y
)

pipe.fit(X_train, y_train)
val_preds = pipe.predict(X_val)

print("VAL accuracy:", round(accuracy_score(y_val, val_preds), 4))
print("VAL macro F1:", round(f1_score(y_val, val_preds, average="macro"), 4))
print("\nValidation report:\n", classification_report(y_val, val_preds))

# Predict on provided test[features]
test_preds = pipe.predict(test[features])
preds_df = pd.DataFrame({"prediction": test_preds})
preds_df.head()



# In[23]:


# Create confusion matrix
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#Build a crosstab
cm_df = pd.crosstab(y_val, val_preds,
                    rownames=['Actual'], colnames=['Predicted'])
print(cm_df)

# Nice plotted CM
cm = confusion_matrix(y_val, val_preds, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.show()



# In[24]:


# Feature importance (run AFTER pipe.fit) 
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

# Ensure the fitted pipeline is in scope
model = pipe  # the fitted Pipeline
check_is_fitted(model)

# Decompose: get the fitted RF and the preprocessor
rf  = model.named_steps["rf"]
pre = model.named_steps["pre"]

# Transformed feature names (after impute/encode)
feat_names = pre.get_feature_names_out()
importances = rf.feature_importances_

# Importance table at the encoded/transformed level
fi_transformed = (
    pd.Series(importances, index=feat_names, name="importance")
      .sort_values(ascending=False)
)
display(fi_transformed.head(20))  # top-20 signal


# In[ ]:




