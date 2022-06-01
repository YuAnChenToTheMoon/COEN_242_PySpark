#Start a new Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime

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

df = df.withColumn("cardPresent", df.cardPresent.cast('int'))
df = df.withColumn("isFraud", df.isFraud.cast('int'))
df = df.withColumn("expirationDateKeyInMatch", df.expirationDateKeyInMatch.cast('int'))



# indexer = StringIndexer(inputCol="merchantName", outputCol = "merchantNameNum")
# df = indexer.setHandleInvalid("skip").fit(df).transform(df)
# indexer = StringIndexer(inputCol="acqCountry", outputCol = "acqCountryNum")
# df = indexer.setHandleInvalid("skip").fit(df).transform(df)
# indexer = StringIndexer(inputCol="merchantCountryCode", outputCol = "merchantCountryCodeNum")
# df = indexer.setHandleInvalid("skip").fit(df).transform(df)
# indexer = StringIndexer(inputCol="merchantCategoryCode", outputCol = "merchantCategoryNum")
# df = indexer.setHandleInvalid("skip").fit(df).transform(df)
# indexer = StringIndexer(inputCol="transactionType", outputCol = "transactionTypeNum")
# df = indexer.setHandleInvalid("skip").fit(df).transform(df)


# In[24]:





# In[26]:


df.columns


# In[28]:


# Create assembler object to include only relevant columns 

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
inputCols=["accountNumber", "customerId", "creditLimit", "availableMoney", "transactionAmount", "cardCVV", "enteredCVV", "cardLast4Digits", "currentBalance", "cardPresent", "expirationDateKeyInMatch"],
outputCol="features")
output = assembler.transform(df)

# In[29]:





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
from pyspark.ml.clustering import KMeans


# In[34]:


#Training the linear model
# lr = LinearRegression(labelCol="availableMoney")
# lr = LogisticRegression(labelCol="isFraud")
# rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=10)
kmeans = KMeans(k=2, seed=0)


# # In[35]:


start_time = datetime.now()
# lrModel = lr.fit(train_data) # for logistic regression
# rfModel = rf.fit(train_data)
kmeans = kmeans.fit(train_data)
duration = (datetime.now() - start_time).total_seconds()

# # In[39]:



# print("Coefficients: {} Intercept: {}".format(lrModel.coefficients,lrModel.intercept))


# # In[40

# Evaluate Kmeans:
from pyspark.ml.evaluation import ClusteringEvaluator
# Evaluate clustering by computing Silhouette score
predictions = kmeans.transform(test_data)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("\n\nSilhouette with squared euclidean distance = " + str(silhouette) + "\n\n")

# Evaluate random forest 
# predictions = rfModel.transform(test_data)
# evaluator = MulticlassClassificationEvaluator(
#     labelCol="isFraud", predictionCol="prediction", metricName="accuracy")
# accuracy = evaluator.evaluate(predictions)
# print("accuracy = %g" % accuracy)

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
# print "\n\n"
# print "execution time:", duration, "seconds"
# print "tp:", tp
# print "tn:", tn
# print "fp:", fp
# print "fn:", fn 
# print "accuracy:", accuracy * 100 , "%"
# print "precision:", precision * 100 , "%"
# print "recall:", recall * 100 , "%"
# print "F1 score:", f1
# print "\n\n"
# final_data.groupBy("isFraud").count().show()

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

