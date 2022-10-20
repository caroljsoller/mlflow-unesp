# Databricks notebook source
# MAGIC %sql
# MAGIC 
# MAGIC select avg(radiant_win) from sandbox_apoiadores.abt_dota_pre_match

# COMMAND ----------

# DBTITLE 1,Imports
#Import das libs

from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
from sklearn import metrics

import mlflow

#import dos dados

sdf = spark.table("sandbox_apoiadores.abt_dota_pre_match")

df = sdf.toPandas()

# COMMAND ----------

#Exibe o uso da memoria no pandas

df.info(memory_usage='deep')

# COMMAND ----------

# DBTITLE 1,Definicao das variaveis
target_column = 'radiant_win'
id_column = 'match_id'

features_columns = list( set(df.columns.tolist()) - set([target_column,id_column]) )
                         
y = df[target_column]
X = df[features_columns]

# COMMAND ----------

# DBTITLE 1,Split Test e Train
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=42)

print("Número de linhas em x_train:", X_train.shape[0])
print("Número de linhas em x_test:", X_test.shape[0])
print("Número de linhas em x_train:", X_train.shape[0])
print("Número de linhas em x_test:", X_test.shape[0])

# COMMAND ----------

from sklearn import tree
from sklearn import ensemble

# model = tree.DecisionTreeClassifier()
model = ensemble.AdaBoostClassifier (n_estimators=100, learning_rate=0.7)
model.fit(X_train, y_train)

# COMMAND ----------

# DBTITLE 1,Setup do Experimento
mlflow.set_experiment("/Users/caroline.soller@unesp.br/dota-unesp-carol")

# COMMAND ----------

# DBTITLE 1,Run do Experimento
with mlflow.start_run():
    
    mlflow.sklearn.autolog()
    
    # model = ensemble.AdaBoostClassifier (n_estimators=100, learning_rate=0.5)
    model = ensemble.RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_train_prob = model.predict_proba(X_train)
    acc_train = metrics.accuracy_score(y_train, y_train_pred)
    print("Acuracia em treino:", acc_train)
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    acc_test = metrics.accuracy_score(y_test, y_test_pred)
    print("Acuracia em teste:", acc_test)
