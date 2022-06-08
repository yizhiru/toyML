package io.github.yizhiru.toyml.util

import org.apache.spark.graphx.VertexId
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

object GraphUtils {

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


  /**
   * 基于用户行为序列构建图，相邻节点构成边
   * 边权重为两顶点共现次数
   */
  def constructGraph(rdd: RDD[(String, Seq[Long])],
                     maxDegree: Int,
                     minOccurrence: Int): RDD[VertexAttr] = {

    val edgeRDD = rdd.flatMapValues { hist =>
      val edge = for (i <- 0 until hist.length - 1)
        yield (hist(i), hist(i + 1))

      edge.filter(t => t._1 != t._2)
    }
      .distinct()

    // 计算边权重
    val weightedEdgeRDD = edgeRDD.map { t => ((t._2._1, t._2._2), 1) }
      .reduceByKey(_ + _)
      .map { t =>
        val weight = t._2
        (t._1._1, (t._1._2, weight))
      }
      .filter(_._2._2 >= minOccurrence)

    // 计算邻居节点
    weightedEdgeRDD.map(t => (t._1, List(t._2)))
      .reduceByKey(_ ::: _)
      .map { t =>
        val arr = t._2.toArray.sortWith { (e1, e2) =>
          e1._2 > e2._2
        }.take(maxDegree)

        VertexAttr(t._1, arr.map(t => (t._1, t._2.toDouble)))
      }
      .cache()
  }

  /**
   * Node2Vec 随机游走策略
   */
  def randomWalk(vertexRdd: RDD[VertexAttr],
                 numWalks: Int,
                 walkLength: Int,
                 p: Double = 1.0,
                 q: Double = 1.0): RDD[Array[VertexId]] = {
    val vertexNeighborhood: RDD[(VertexId, Array[(VertexId, Double)])] = vertexRdd.map(vn => (vn.vertexId, vn.neighbors))
      .cache()

    // 随机游走节点路径的所有迭代汇总
    var walkPath: RDD[Array[VertexId]] = null
    for (_ <- 0 until numWalks) {
      // 单次沿边(t, v)随机游走路径，源顶点为t，目的顶点为v
      // 数据格式：(目的顶点v, (边信息, 游走路径))
      var perWalkPath: RDD[(VertexId, (EdgeAttr, ArrayBuffer[VertexId]))] = vertexRdd.map { vn =>
        (vn.vertexId,
          (EdgeAttr(Array.empty, vn.neighbors), ArrayBuffer(vn.vertexId)))
      }

      for (_ <- 1 until walkLength) {
        perWalkPath = perWalkPath.map { t =>
          val path = t._2._2
          val source: VertexId = if (path.length > 1) path.last else -1L
          val next: VertexId = nextVertex(source, t._2._1, p, q)
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

}
