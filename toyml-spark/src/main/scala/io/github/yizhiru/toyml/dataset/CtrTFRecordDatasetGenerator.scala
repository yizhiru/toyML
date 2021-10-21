package io.github.yizhiru.toyml.dataset

import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, udf}
import scopt.OptionParser

object CtrTFRecordDatasetGenerator {


  case class Params(
                     inputTable: String = null,
                     outputPath: String = null,
                     bizDay: String = null)

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("CtrTFRecordDatasetGenerator") {
      head("convert hive table to tfrecord")
      opt[String]("inputTable")
        .text("input table")
        .required()
        .action((x, c) => c.copy(inputTable = x))
      opt[String]("outputPath")
        .text("outputPath")
        .required()
        .action((x, c) => c.copy(outputPath = x))
      opt[String]("bizDay")
        .text("bizDay")
        .required()
        .action((x, c) => c.copy(bizDay = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val spark = SparkSession.builder()
      .appName(this.getClass.getName)
      .enableHiveSupport()
      .getOrCreate()


    val df = spark.sql(s"select * from ${params.inputTable} where pt = '${params.bizDay}' ")

    // split string to int array
    val splitUDF = udf { hist: String =>
      hist.split(",")
        .map(e => e.toInt)
    }

    val resultDF = df.withColumn("click_item_hist", splitUDF(col("click_item_hist")))
      .withColumn("click_seller_hist", splitUDF(col("click_seller_hist")))
      .withColumn("click_cid4_hist", splitUDF(col("click_cid4_hist")))


    val reservedColNames = Seq("uid", "item_id", "is_click", "hash_buyer_id",
      "click_item_hist", "click_seller_hist")
    resultDF.select(reservedColNames.head, reservedColNames.tail: _*)
      .repartition(1)
      .write
      .mode(SaveMode.Overwrite)
      .format("tfrecords")
      .option("recordType", "Example")
      .option("codec", "org.apache.hadoop.io.compress.GzipCodec")
      .save(params.outputPath)

    spark.stop()

  }

}
