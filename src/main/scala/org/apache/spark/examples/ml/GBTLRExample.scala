package org.apache.spark.examples.ml

import org.apache.spark.ml.gbtlr.GBTLRClassifier
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.sql.SparkSession

// scalastyle:off println


object GBTLRExample {
  def main(args: Array[String]): Unit = {
    if (args.length < 1) {
      // scalastyle:off
      println("Usage: program input_path")
      sys.exit(1)
    }
    val spark = SparkSession
        .builder()
        //.master("local[2]")
  //.master("spark://10.203.3.95:7077")
        .appName("gbtlr example")
        .getOrCreate()

    val startTime = System.currentTimeMillis()

    val inputPath = args(0)
    //val inputPath = "/Users/oyo/Downloads/develop/spark-gbtlr/data/bank/bank-full.csv"
    val dataset = spark.read.option("header", "true").option("inferSchema", "true")
        .option("delimiter", ";").csv(inputPath) //"data/bank/bank-full.csv")

    val columnNames = Array("job", "marital", "education",
      "default", "housing", "loan", "contact", "month", "poutcome", "y")
    val indexers = columnNames.map(name => new StringIndexer()
        .setInputCol(name).setOutputCol(name + "_index"))
    val pipeline = new Pipeline().setStages(indexers)
    val data1 = pipeline.fit(dataset).transform(dataset)
    val data2 = data1.withColumnRenamed("y_index", "label")

    val assembler = new VectorAssembler()
    assembler.setInputCols(Array("age", "job_index", "marital_index",
      "education_index", "default_index", "balance", "housing_index",
      "loan_index", "contact_index", "day", "month_index", "duration",
      "campaign", "pdays", "previous", "poutcome_index"))
    assembler.setOutputCol("features")

    val data3 = assembler.transform(data2)
    val data4 = data3.randomSplit(Array(4, 1))

    val gBTLRClassifier = new GBTLRClassifier()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setGBTMaxIter(10)
        .setLRMaxIter(100)
        .setRegParam(0.01)
        .setElasticNetParam(0.5)

   /* val predictData = data3.select("age", "job_index", "marital_index",
      "education_index", "default_index", "balance", "housing_index",
      "loan_index", "contact_index", "day", "month_index", "duration",
      "campaign", "pdays", "previous", "poutcome_index")*/
    val model = gBTLRClassifier.fit(data4(0))
    val summary = model.evaluate(data4(1))
    val endTime = System.currentTimeMillis()
    val auc = summary.binaryLogisticRegressionSummary
        .asInstanceOf[BinaryLogisticRegressionSummary].areaUnderROC

    model.write.overwrite().save("/bigdata_ai/test/gbdtlr.model")





    //summary.binaryLogisticRegressionSummary.predictions
    //val predic = model.predict()

  println(data3.show(false))

    //println(s"Model's Param:\n" + model.gbtModel.treeWeights)

    //val test =  Vectors.dense(data3.select("features").map(row=>model.predict())

    val b = data3.select("features").rdd
    b.map(row=>println(row.get(1).toString() + row.get(2).toString()))

    val test = Vectors.dense(0,1,3,5,8,9,11,12,13,58.0,1.0,1.0,2143.0,1.0,5.0,261.0,1.0)

    print("nihaoVector:\n" + data3.select("label").show(1))
    print(s"MOdel's Predict:\n" + model.predict(test)+"\n")
    println(s"Training and evaluating cost ${(endTime - startTime) / 1000} seconds")
    println(s"The model's auc: ${auc}")
  }
}

// scalastyle:on println

