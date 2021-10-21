package io.github.yizhiru.toyml.dataset

import org.apache.spark.sql.{SaveMode, SparkSession}
import scopt.OptionParser

object RerankingTFRecordDataSetGenerator {

  case class Params(
                     inputTable: String = null,
                     outputPath: String = null,
                     bizDay: String = null
                   )

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("RerankingTFRecordDataSetGenerator") {
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

    val rawDF = spark.sql(s"select * from ${params.inputTable} where pt = '${params.bizDay}' ")

    import spark.implicits._
    val df = rawDF.rdd
      .map { row =>
        val qid = row.getInt(0)
        val channel = row.getString(1)
        val uid = row.getString(2)
        val itemId = row.getAs[Seq[String]](7).map(ele => Array(ele)).toArray
        val label = row.getAs[Seq[Int]](72).map(ele => Array(ele)).toArray
        val exampleListSize = itemId.length

        (qid, channel, uid, itemId, label, exampleListSize)
      }.toDF("qid", "channel", "uid", "item_id", "label", "example_list_size")

    df.repartition(1)
      .write
      .mode(SaveMode.Overwrite)
      .format("tfrecords")
      .option("recordType", "SequenceExample")
      .option("codec", "org.apache.hadoop.io.compress.GzipCodec")
      .save(params.outputPath)
    spark.stop()
  }
}
