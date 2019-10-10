package org.apache.spark.examples.ml

import org.apache.spark.ml
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.{Pipeline, gbtlr}
import org.apache.spark.ml.gbtlr.{GBTLRClassificationModel, GBTLRClassifier}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object GBTLRModeTest {
 def main(args: Array[String]): Unit = {
   if (args.length < 1) {
     // scalastyle:off
     println("Usage: program input_path")
     sys.exit(1)
   }

   val inputPath = args(0)

   val spark = SparkSession
     .builder()
     .appName("gbtlr example")
     .getOrCreate()
   val dataset = spark.read.option("header", "true").option("inferSchema", "true")
     .option("delimiter", ";").csv(inputPath) //"data/bank/bank-full.csv")

   val columnNames = Array("job", "marital", "education",
     "default", "housing", "loan", "contact", "month", "poutcome")
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

   val modelpath = args(1)

   val gBTLRClassifier = GBTLRClassificationModel.load(modelpath) //" /bigdata_ai/test/gbdtlr.model")

   //gBTLRClassifier.evaluate(data3)

   val test = Vectors.dense(0, 1, 3, 5, 8, 9, 11, 12, 13, 58.0, 1.0, 1.0, 2143.0, 1.0, 5.0, 261.0, 1.0)


   val b = data3.select("features").rdd
   //b.map(row=>row.get(1).toString() + row.get(2).toString())
   val d = b.map(row=> Vectors.dense(row.get(0).toString().replace("[","").replace("]","").replace("(","").replace(")","").split(",").map(_.toDouble)))
   //val c = b.first().get(0).toString().replace("[","").replace("]","").replace("(","").replace(")","").split(",")
   //val select = d.map(row=>row.toArray.toList).map(row=>(row(2),row(3),row(4),row(5),row(6),row(7),row(8),row(9),row(10),row(11),row(12),row(13),row(14),row(15),row(16),row(17)))
   val select1 = d.map(row=>row.toArray.toList).map(row=>row(2)+","+row(3)+","+row(4)+","+row(5)+","+row(6)+","+row(7)+","+row(8)+","+row(9)+","+row(10)+","+row(11)+","+row(12)+","+row(13)+","+row(14)+","+row(15)+","+row(16)+","+row(17))

   val lastfeature = select1.map(row=>Vectors.dense(row.split(",").map(_.toDouble)))

   println(d.take(2).tail.toArray.mkString(","))
   //println(select.first().toString())
   //print("nihaoVector:\n" + c.toArray.length + "\n" )
   //println(c.mkString(" & "))
   print(s"MOdel's Predict:\n" + gBTLRClassifier.predict(test) + "\n")
   println(s"nihaoyuce:\n" + + gBTLRClassifier.predict(lastfeature.first()) + "\n")

 }

}


