package io.github.yizhiru.toyml.util

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution
import spire.ClassTag

object RandomUtils {

  /**
   * numpy.random.choice的scala实现，根据概率随机抽样样本
   *
   * @param arr   样本数组
   * @param probs 概率数组，表示每一个样本的抽样概率，概率之和为1
   * @tparam V 泛型
   * @return 一次随机抽样的样本
   */
  def randomChoice[V: ClassTag](arr: Array[V],
                                probs: Array[Double]): V = {
    val dist = new EnumeratedIntegerDistribution(arr.indices.toArray, probs)
    arr(dist.sample())
  }

}
