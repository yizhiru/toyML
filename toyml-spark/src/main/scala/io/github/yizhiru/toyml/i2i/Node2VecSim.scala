package io.github.yizhiru.toyml.i2i

import io.github.yizhiru.toyml.util.GraphUtils
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import scopt.OptionParser


/**
 * Node2Vec 算法实现，参考论文 https://arxiv.org/abs/1607.00653
 *
 * 基于用户行为序列构造图
 */
object Node2VecSim {

  case class Params(
                     inputTable: String = null,
                     outputTable: String = null,
                     minOccurrence: Int = 2,
                     numWalks: Int = 10,
                     walkLength: Int = 20,
                     p: Double = 1.0,
                     q: Double = 1.0,
                     maxDegree: Int = 200,
                     lr: Double = 0.025,
                     numPartitions: Int = 8,
                     numIter: Int = 10,
                     windowSize: Int = 5,
                     nodeDim: Int = 128,
                     numSimNode: Int = 50
                   )


  def main(args: Array[String]): Unit = {
    val defaultParams = Params()

    val parser = new OptionParser[Params]("Node2Vec") {
      head("Node2Vec")
      opt[String]("inputTable")
        .text("inputTable")
        .required()
        .action((x, c) => c.copy(inputTable = x))
      opt[String]("outputTable")
        .text("outputTable")
        .required()
        .action((x, c) => c.copy(outputTable = x))
      opt[Int]("minOccurrence")
        .text("the minimum occurrence between two nodes in user behaviour sequence")
        .action((x, c) => c.copy(minOccurrence = x))
      opt[Int]("numWalks")
        .text("the number of random walk in graph ")
        .action((x, c) => c.copy(numWalks = x))
      opt[Int]("walkLength")
        .text("the length of random walk")
        .action((x, c) => c.copy(walkLength = x))
      opt[Double]("p")
        .text("return parameter")
        .action((x, c) => c.copy(p = x))
      opt[Double]("q")
        .text("in-out parameter")
        .action((x, c) => c.copy(q = x))
      opt[Int]("maxDegree")
        .text("maxDegree")
        .action((x, c) => c.copy(maxDegree = x))
      opt[Double]("lr")
        .text("lr")
        .action((x, c) => c.copy(lr = x))
      opt[Int]("numPartitions")
        .text("num partition of word2vec and similarity computation")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Int]("numIter")
        .text("num Iteration of word2vec")
        .action((x, c) => c.copy(numIter = x))
      opt[Int]("windowSize")
        .text("windowSize")
        .action((x, c) => c.copy(windowSize = x))
      opt[Int]("nodeDim")
        .text("nodeDim")
        .action((x, c) => c.copy(nodeDim = x))
      opt[Int]("numSimNode")
        .text("numSimNode")
        .action((x, c) => c.copy(numSimNode = x))

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

    val rdd = spark.sql(s"select distinct user_id, item_hist from ${params.inputTable}")
      .rdd
      .map { row =>
        val uid = row.getAs[String]("user_id")
        val itemHist = row.getAs[Seq[Long]]("item_hist")
        (uid, itemHist)
      }

    val vertexRdd = GraphUtils.constructGraph(rdd, params.maxDegree, params.minOccurrence)
    val walkPath = GraphUtils.randomWalk(vertexRdd,
      params.numWalks,
      params.walkLength,
      params.p,
      params.q)

    findSimNode(spark,
      walkPath.map(t => t.map(_.toString).toSeq),
      params)

    spark.stop()
  }

  private def cosineSimilarity(a: Array[Float], b: Array[Float]): Double = {
    var dot = 0.0
    var normA = 0.0
    var normB = 0.0
    var i = 0
    while (i < a.length) {
      dot += a(i) * b(i)
      normA += a(i) * a(i)
      normB += b(i) * b(i)
      i += 1
    }
    if (normA == 0.0 || normB == 0.0) 0.0 else dot / (math.sqrt(normA) * math.sqrt(normB))
  }

  /**
   * Skip-gram 训练顶点向量，分布式计算最相似节点 top-n
   */
  private def findSimNode(spark: SparkSession,
                          walkPath: RDD[Seq[String]],
                          params: Params) = {
    val word2vec = new Word2Vec()
      .setLearningRate(params.lr)
      .setNumIterations(params.numIter)
      .setWindowSize(params.windowSize)
      .setMinCount(0)
      .setVectorSize(params.nodeDim)
      .setNumPartitions(params.numPartitions)

    val model = word2vec.fit(walkPath)
    val vectors = model.getVectors.toArray
    val vectorsBC = spark.sparkContext.broadcast(vectors)

    import spark.implicits._
    val simRDD = spark.sparkContext
      .parallelize(vectors.map(_._1), params.numPartitions)
      .flatMap { itemId =>
        val vecMap = vectorsBC.value.toMap
        val queryVec = vecMap(itemId)
        vectorsBC.value
          .filter(_._1 != itemId)
          .map { case (id2, vec2) => (id2, cosineSimilarity(queryVec, vec2)) }
          .sortBy(-_._2)
          .take(params.numSimNode)
          .map { case (id2, score) => (itemId, id2, score) }
      }

    val simDF = simRDD.toDF("item_id1", "item_id2", "sim_score")

    val tempViewName = s"temp_node2vec_recs"
    simDF.createOrReplaceTempView(tempViewName)
    spark.sql(s"drop table if exists ${params.outputTable}")
    spark.sql(s"create table ${params.outputTable} stored as orc as select * from $tempViewName")
  }

}
