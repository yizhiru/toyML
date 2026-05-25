package io.github.yizhiru.toyml.dataset

import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.sql.functions.{col, split}
import scopt.OptionParser

object CtrTFRecordDatasetGenerator {


  case class Params(
                     inputTable: String = null,
                     outputPath: String = null,
                     bizDay: String = null,
                     numPartitions: Int = 8)

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
      opt[Int]("numPartitions")
        .text("number of output partitions")
        .action((x, c) => c.copy(numPartitions = x))
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

    val resultDF = df
      .withColumn("click_item_hist", split(col("click_item_hist"), ",").cast("array<int>"))
      .withColumn("click_seller_hist", split(col("click_seller_hist"), ",").cast("array<int>"))
      .withColumn("click_cid4_hist", split(col("click_cid4_hist"), ",").cast("array<int>"))


    val reservedColNames = Seq("uid", "item_id", "is_click", "hash_buyer_id",
      "click_item_hist", "click_seller_hist", "click_cid4_hist")
    resultDF.select(reservedColNames.head, reservedColNames.tail: _*)
      .repartition(params.numPartitions)
      .write
      .mode(SaveMode.Overwrite)
      .format("tfrecords")
      .option("recordType", "Example")
      .option("codec", "org.apache.hadoop.io.compress.GzipCodec")
      .save(params.outputPath)

    spark.stop()

  }

}
