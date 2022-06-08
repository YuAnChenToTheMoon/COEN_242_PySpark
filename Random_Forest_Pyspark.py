#Start a new Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from datetime import datetime

spark = SparkSession.builder.appName('random_forest').getOrCreate()


#Read the csv file in a dataframe
df = spark.read.csv('transactions.csv',inferSchema=True,header=True)



#Convert string categorical values to integer categorical values
from pyspark.ml.feature import StringIndexer

df = df.withColumn("cardPresent", df.cardPresent.cast('int'))
df = df.withColumn("isFraud", df.isFraud.cast('int'))
df = df.withColumn("expirationDateKeyInMatch", df.expirationDateKeyInMatch.cast('int'))


# Feature selection
# Create assembler object to include only relevant columns 

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(
inputCols=["accountNumber", "customerId", "creditLimit", "availableMoney", "transactionAmount", "cardCVV", "enteredCVV", "cardLast4Digits", "currentBalance", "cardPresent", "expirationDateKeyInMatch"],
outputCol="features")
output = assembler.transform(df)


# Balance Data
final_data = output.select("features", "isFraud")
valid_data = final_data.filter(df.isFraud == 0)
valid_data = valid_data.orderBy(F.rand()).limit(12417)
invalid_data = final_data.filter(df.isFraud == 1)
final_data = valid_data.union(invalid_data)


#Split the train and test data into 70/30 ratio
train_data,test_data = final_data.randomSplit([0.70,0.30])


from pyspark.ml.classification import RandomForestClassifier


# Training the model
rf = RandomForestClassifier(labelCol="isFraud", featuresCol="features", numTrees=10)



start_time = datetime.now()
rfModel = rf.fit(train_data)

# calculate the execution time
duration = (datetime.now() - start_time).total_seconds()


# Evaluate random forest 
predictions = rfModel.transform(test_data)
evaluator = MulticlassClassificationEvaluator(
    labelCol="isFraud", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("accuracy = %g" % accuracy)
