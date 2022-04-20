package io.github.yizhiru.toyml.i2i

import io.github.yizhiru.toyml.util.RandomUtils
import org.apache.spark.graphx.VertexId
import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.sql.SparkSession
import org.roaringbitmap.RoaringBitmap
import scopt.OptionParser

import scala.collection.mutable.ArrayBuffer


/**
 * Node2Vec 算法实现，参考论文 https://arxiv.org/abs/1607.00653
 * 边的权重计算point-wise互信息。
 */
object Node2VecSim {

  case class Params(
                     inputTable: String = null,
                     outputTable: String = null,
                     numWalks: Int = 10,
                     walkLength: Int = 20,
                     p: Double = 1.0,
                     q: Double = 1.0,
                     maxDegree: Int = 200,
                     lr: Double = 0.025,
                     numIter: Int = 10,
                     windowSize: Int = 5,
                     nodeDim: Int = 128,
                     numSimNode: Int = 50
                   )

  /**
   * 顶点信息，包括：顶点id及边权重
   *
   * @param neighbors 顶点的邻居顶点
   */
  case class VertexAttr(vertexId: VertexId,
                        neighbors: Array[(VertexId, Double)] = Array.empty) extends Serializable


  /**
   * 边的两个顶点的邻居信息，包括：顶点id及边权重
   *
   * @param sourceNeighbors 源顶点的邻居信息
   * @param destNeighbors   目的顶点的邻居信息
   */
  case class EdgeAttr(sourceNeighbors: Array[VertexId] = Array.empty,
                      destNeighbors: Array[(VertexId, Double)] = Array.empty) extends Serializable


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

    val rdd = spark.sql(s"select distinct user_id, item_id from ${params.inputTable}")
      .rdd
      .map { row =>
        val uid = row.getAs[String]("user_id")
        val itemId = row.getAs[String]("item_id").toLong
        (uid, itemId)
      }

    val vertexNeighborhood = collectNeighborhood(rdd, params.maxDegree)
    val walkPath = randomWalk(vertexNeighborhood, params)

    findSimNode(spark,
      walkPath.map(t => t.map(_.toString).toSeq),
      params)

    spark.stop()
  }

  /**
   * 计算邻居节点，基于point-wise互信息计算i2i边权重2
   * pmi = log(nab) - log(na) - log(nb) + log(D)
   * weight = nab * pmi
   * nab 为同时点击a、b的用户数，na为点击a的用户数，nb为点击b的用户数，D为所有点击行为数
   */
  private def collectNeighborhood(rdd: RDD[(String, Long)],
                                  maxDegree: Int): RDD[VertexAttr] = {

    val dBC = rdd.sparkContext.broadcast(rdd.count())

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

    // 通过单个用户关联，创建候选i2i
    val i2iRDD = rdd.join(rdd)
      .map(_._2)
      .filter(t => t._1 < t._2)
      .distinct()
      .join(item2BitMapRDD)
      .map(t => (t._2._1, (t._1, t._2._2)))
      .join(item2BitMapRDD)
      .map(t => (t._2._1._1, t._1, t._2._1._2, t._2._2))

    // 计算边权重
    val edgeWeight: RDD[(VertexId, (VertexId, Double))] = i2iRDD.flatMap { t =>
      val na = t._3.getCardinality
      val nb = t._4.getCardinality
      val andBitMap = RoaringBitmap.and(t._3, t._4)
      val nab = andBitMap.getCardinality
      val weight = nab * (Math.log(nab) - Math.log(na) - Math.log(nb) + Math.log(dBC.value))
      // 构建双边
      Array((t._1, (t._2, weight)), (t._2, (t._1, weight)))
    }

    // 计算邻居节点
    edgeWeight.map(t => (t._1, List(t._2)))
      .reduceByKey(_ ::: _)
      .map { t =>
        val arr = t._2.toArray.sortWith { (e1, e2) =>
          e1._2 > e2._2
        }.take(maxDegree)

        VertexAttr(t._1, arr)
      }
      .cache()
  }

  /**
   * 沿边(t, v) 随机游走，biased参数alpha_pq(t,x) 计算公式参见原论文：
   * 如果x=t，则alpha_pq(t,x)=1/p；
   * 如果x为t的邻居顶点，则alpha_pq(t,x)=1；
   * 除上述两种情况外，则alpha_pq(t,x)=1/q；
   * 计算转移概率，根据转移概率随机游走到下一顶点。
   *
   * @param sourceVertex 边(t, v)源顶点
   * @param edgeAttr     边(t, v)顶点的邻居信息
   * @param p            随机游走超参
   * @param q            随机游走超参
   */
  private def nextVertex(sourceVertex: VertexId,
                         edgeAttr: EdgeAttr,
                         p: Double,
                         q: Double): VertexId = {
    val biasWeight = edgeAttr.destNeighbors.map { t =>
      val weight = if (t._1 == sourceVertex) t._2 / p
      else if (edgeAttr.sourceNeighbors.toSet.contains(t._1)) t._2
      else t._2 / q

      (t._1, weight)
    }
    val neighbors = biasWeight.map(_._1)
    val weightSum = biasWeight.map(_._2).sum
    val transitionProb = biasWeight.map(t => t._2 / weightSum)
    RandomUtils.randomChoice(neighbors, transitionProb)
  }

  /**
   * Node2Vec 随机游走策略
   *
   * @param vertexRdd 图的顶点
   * @param params    相关参数
   */
  private def randomWalk(vertexRdd: RDD[VertexAttr],
                         params: Params): RDD[Array[VertexId]] = {
    val vertexNeighborhood: RDD[(VertexId, Array[(VertexId, Double)])] = vertexRdd.map(vn => (vn.vertexId, vn.neighbors))
      .cache()

    // 随机游走节点路径的所有迭代汇总
    var walkPath: RDD[Array[VertexId]] = null
    for (_ <- 0 until params.numWalks) {
      // 单次沿边(t, v)随机游走路径，源顶点为t，目的顶点为v
      // 数据格式：(目的顶点v, (边信息, 游走路径))
      var perWalkPath: RDD[(VertexId, (EdgeAttr, ArrayBuffer[VertexId]))] = vertexRdd.map { vn =>
        (vn.vertexId,
          (EdgeAttr(Array.empty, vn.neighbors), ArrayBuffer(vn.vertexId)))
      }

      for (_ <- 1 until params.walkLength) {
        perWalkPath = perWalkPath.map { t =>
          val path = t._2._2
          val source: VertexId = if (path.length > 1) path.last else -1L
          val next: VertexId = nextVertex(source, t._2._1, params.p, params.q)
          path.append(next)

          (next, (EdgeAttr(t._2._1.destNeighbors.map(_._1), Array.empty), path))
        }

        perWalkPath = perWalkPath.join(vertexNeighborhood)
          .map { t =>
            (t._1, (EdgeAttr(t._2._1._1.sourceNeighbors, t._2._2), t._2._1._2))
          }
          .cache()
      }

      if (walkPath != null) {
        walkPath = walkPath.union(perWalkPath.map(_._2._2.toArray)).cache()
        perWalkPath.unpersist(blocking = false)
      } else {
        walkPath = perWalkPath.map(_._2._2.toArray)
      }
    }

    walkPath
  }

  /**
   * Skip-gram 训练顶点向量，寻找最相似节点top-n
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

    val model = word2vec.fit(walkPath)

    import spark.implicits._

    val simDF = model.getVectors
      .keys
      .flatMap { itemId =>
        val simNodes = model.findSynonyms(itemId, params.numSimNode)
        simNodes.map(t => (itemId, t._1, t._2))
      }
      .toSeq
      .toDF("item_id1", "item_id2", "sim_score")

    val tempViewName = s"temp_node2vec_recs"
    simDF.createOrReplaceTempView(tempViewName)
    spark.sql(s"use temp")
    spark.sql(s"drop table if exists ${params.outputTable}")
    spark.sql(s"create table ${params.outputTable} stored as orc as select * from $tempViewName")
  }

}
