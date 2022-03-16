package io.github.yizhiru.toyml.i2i

import org.apache.spark.sql.SparkSession
import org.roaringbitmap.RoaringBitmap
import scopt.OptionParser

/**
 * itemCF 算法Bitmap实现版本，可用于生产环境
 */
object ItemCFSim {

  case class Params(
                     inputTable: String = null,
                     outputTable: String = null
                   )

  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("itemCF") {
      head("itemCF")
      opt[String]("inputTable")
        .text("input table")
        .required()
        .action((x, c) => c.copy(inputTable = x))
      opt[String]("outputTable")
        .text("output table")
        .required()
        .action((x, c) => c.copy(outputTable = x))
    }

    parser.parse(args, defaultParams) match {
      case Some(params) => run(params)
      case _ => sys.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val spark = SparkSession
      .builder()
      .appName(this.getClass.getName)
      .enableHiveSupport()
      .getOrCreate()

    import spark.implicits._

    val df = spark.sql(s"select distinct uid, item_id from ${params.inputTable}")
    val rdd = df.rdd
      .map { r =>
        val userId = r.getAs[String]("uid")
        val itemId = r.getAs[String]("item_id")

        (userId, itemId)
      }
    rdd.persist()

    // 对用户做编号
    val user2IndexRDD = rdd.map(_._1)
      .distinct()
      .zipWithIndex()
      .mapValues(_.toInt)

    val item2BitMapRDD = rdd.join(user2IndexRDD)
      .map(t => (t._2._1, t._2._2))
      .groupByKey()
      .map { t =>
        val bitmap = new RoaringBitmap()
        t._2.foreach { uIndex =>
          bitmap.add(uIndex)
        }
        (t._1, bitmap)
      }

    // 通过单个用户关联，创建候选i2i数据集
    val i2iRDD = rdd.join(rdd)
      .map(_._2)
      .filter(t => t._1 < t._2)
      .distinct()
      .join(item2BitMapRDD)
      .map(t => (t._2._1, (t._1, t._2._2)))
      .join(item2BitMapRDD)
      .map(t => (t._2._1._1, t._1, t._2._1._2, t._2._2))

    // 计算相似度
    val simDF = i2iRDD.map { t =>
      val na = t._3.getCardinality
      val nb = t._4.getCardinality
      val andBitMap = RoaringBitmap.and(t._3, t._4)
      val nab = andBitMap.getCardinality
      val score = nab / Math.sqrt(na * nb)
      (t._1, t._2, score, na, nb, nab)
    }
      .toDF("item_id1", "item_id2", "sim_score", "na", "nb", "nab")
    simDF.show()


    val tempViewName = s"temp_item_cf_recs"
    simDF.createOrReplaceTempView(tempViewName)

    spark.sql(s"drop table if exists ${params.outputTable}")
    spark.sql(s"create table ${params.outputTable} stored as orc as select * from $tempViewName")

    spark.stop()

  }

}
