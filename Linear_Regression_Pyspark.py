#Start a new Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# App named 'Cruise'
spark = SparkSession.builder.appName('cruise').getOrCreate()


# In[6]:


#Read the csv file in a dataframe
df = spark.read.csv('transactions.csv',inferSchema=True,header=True)


# In[7]:


#Check the structure of schema
df.printSchema()


# In[8]:


df.show()


# In[9]:


df.describe().show()


# In[10]:


# df.groupBy('Cruise_line').count().show()


# In[23]:


#Convert string categorical values to integer categorical values
from pyspark.ml.feature import StringIndexer
# indexer = StringIndexer(inputCol="merchantName", outputCol = "merchantNameNum")
# indexed = indexer.setHandleInvalid("skip").fit(df).transform(df)
# indexer = StringIndexer(inputCol="acqCountry", outputCol = "acqCountryNum")
# indexed = indexer.setHandleInvalid("skip").fit(indexed).transform(indexed)
# indexer = StringIndexer(inputCol="merchantCountryCode", outputCol = "merchantCountryCodeNum")
# indexed = indexer.setHandleInvalid("skip").fit(indexed).transform(indexed)
# indexer = StringIndexer(inputCol="merchantCategoryCode", outputCol = "merchantCategoryNum")
# indexed = indexer.setHandleInvalid("skip").fit(indexed).transform(indexed)
# indexer = StringIndexer(inputCol="transactionType", outputCol = "transactionTypeNum")
# indexed = indexer.setHandleInvalid("skip").fit(indexed).transform(indexed)
indexed = df.withColumn("cardPresent", df.cardPresent.cast('int'))
indexed = indexed.withColumn("isFraud", indexed.isFraud.cast('int'))
indexed = indexed.withColumn("expirationDateKeyInMatch", indexed.expirationDateKeyInMatch.cast('int'))


# In[24]:


from pyspark.ml.linalg import Vectors


# In[25]:


from pyspark.ml.feature import VectorAssembler


# In[26]:


indexed.columns


# In[28]:


# Create assembler object to include only relevant columns 
assembler = VectorAssembler(
inputCols=["accountNumber", "customerId", "creditLimit", "availableMoney", "transactionAmount", "cardCVV", "enteredCVV", "cardLast4Digits", "currentBalance", "cardPresent", "expirationDateKeyInMatch"],
outputCol="features")


# In[29]:


output = assembler.transform(indexed)


# In[30]:


# output.select("features","availableMoney").show()


# In[31]:

final_data = output.select("features", "isFraud")
valid_data = final_data.filter(df.isFraud == 0)
valid_data = valid_data.orderBy(F.rand()).limit(12417)
invalid_data = final_data.filter(df.isFraud == 1)
final_data = valid_data.union(invalid_data)

# In[32]:


#Split the train and test data into 70/30 ratio
train_data,test_data = final_data.randomSplit([0.70,0.30])


# In[33]:


from pyspark.ml.regression import LinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier


# In[34]:


#Training the linear model
# lr = LinearRegression(labelCol="availableMoney")
# lr = LogisticRegression(labelCol="isFraud")
rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=10)


# # In[35]:



# lrModel = lr.fit(train_data) # for logistic regression
rfModel = rf.fit(train_data)


# # In[39]:



# print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# # In[40

# Evaluate random forest 
predictions = rfModel.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="isFraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("accuracy = %g" % accuracy)

# #Evaluate the results with the test data for logistic regression
# test_results = lrModel.evaluate(test_data)
# test_result = test_results.predictions
# test_result.filter(test_result["isFraud"] == 1).show(5)

# tp = test_result[(test_result.isFraud == 1) & (test_result.prediction == 1)].count()
# tn = test_result[(test_result.isFraud == 0) & (test_result.prediction == 0)].count()
# fp = test_result[(test_result.isFraud == 0) & (test_result.prediction == 1)].count()
# fn = test_result[(test_result.isFraud == 1) & (test_result.prediction == 0)].count()
# accuracy = float(tp + tn) / (tp + tn + fp + fn)
# precision = float(tp) / (tp + fp)
# recall = float(tp) / (tp + fn)
# f1 = 2*(recall*precision)/(recall+precision)
# final_data.groupBy("isFraud").count().show()
# print("tp:", tp)
# print("tn:", tn)
# print("fp:", fp)
# print("fn:", fn)
# print("accuracy:", accuracy)
# print("precision:", precision)
# print("recall:", recall)
# print("F1 score:", f1)

# # In[41]:


# print("RMSE: {}".format(test_results.rootMeanSquaredError))
# print("MSE: {}".format(test_results.meanSquaredError))
# print("R2: {}".format(test_results.r2))



# # In[42]:


# from pyspark.sql.functions import corr


# # In[43]:


# #Checking for correlations to explain high R2 values
# df.select(corr('creditLimit','availableMoney')).show()


# # In[44]:


# df.select(corr('creditLimit','transactionAmount')).show()

