#Start a new Spark Session
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from datetime import datetime

spark = SparkSession.builder.appName('logistic').getOrCreate()


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


from pyspark.ml.classification import LogisticRegression


# Training the model
lr = LogisticRegression(labelCol="isFraud")
start_time = datetime.now()
lrModel = lr.fit(train_data)

# calculate execution time
duration = (datetime.now() - start_time).total_seconds()


# Evaluate the results with the test data for logistic regression
test_results = lrModel.evaluate(test_data)
test_result = test_results.predictions
test_result.filter(test_result["isFraud"] == 1).show(5)

tp = test_result[(test_result.isFraud == 1) & (test_result.prediction == 1)].count()
tn = test_result[(test_result.isFraud == 0) & (test_result.prediction == 0)].count()
fp = test_result[(test_result.isFraud == 0) & (test_result.prediction == 1)].count()
fn = test_result[(test_result.isFraud == 1) & (test_result.prediction == 0)].count()
accuracy = float(tp + tn) / (tp + tn + fp + fn)
precision = float(tp) / (tp + fp)
recall = float(tp) / (tp + fn)
f1 = 2*(recall*precision)/(recall+precision)
print "\n\n"
print "execution time:", duration, "seconds"
print "tp:", tp
print "tn:", tn
print "fp:", fp
print "fn:", fn 
print "accuracy:", accuracy * 100 , "%"
print "precision:", precision * 100 , "%"
print "recall:", recall * 100 , "%"
print "F1 score:", f1
print "\n\n"
final_data.groupBy("isFraud").count().show()
