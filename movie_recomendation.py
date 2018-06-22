import json
import math

from pyspark import SparkConf
from pyspark.mllib.recommendation import ALS
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import *

spark = SparkSession \
        .builder \
        .appName("movierecom") \
        .getOrCreate()
        
conf = SparkConf()

conf.setMaster('give the spark cluster path here')\
    .setAppName('test')\
    .set("spark.executor.memory", '8g')\
    .set('spark.executor.cores', '160')\
    .set('spark.cores.max', '160')\
    .set("spark.driver.memory",'8g')        
   
sc = spark.sparkContext

lines = sc.textFile('ratings.csv')

line1 = lines.first()

ratings = lines.filter(lambda x:x not in line1)\
               .map(lambda x:x.split(',')[:-1])\
               .map(lambda x:(x[0],x[1],x[2])).cache()

m = sc.textFile('movies.csv')

m1 = m.first()

movies = m.filter(lambda x:x!=m1)\
          .map(lambda x:x.split(','))\
          .map(lambda x:(x[0],x[1])).cache()

training_RDD, validation_RDD, test_RDD = ratings.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank
