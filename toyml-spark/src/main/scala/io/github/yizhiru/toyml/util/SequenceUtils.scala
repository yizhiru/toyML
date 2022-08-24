package io.github.yizhiru.toyml.util

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

object SequenceUtils {

  /**
   * word2vec的skip-gram模型实现，参考tf.keras代码
   *
   * @param sequence        句子输入序列，为词的索引值
   * @param vocabularySize  词典大小
   * @param windowSize      上下文窗口大小
   * @param negativeSamples 负样本采样比，为正样本数量的倍数
   */
  def skipGram(sequence: Array[Int],
               vocabularySize: Int,
               windowSize: Int = 4,
               negativeSamples: Double = 1.0): (Array[(Int, Int)], Array[Int]) = {
    val couples = ArrayBuffer[(Int, Int)]()
    val labels = ArrayBuffer[Int]()
    val seqLen = sequence.length
    for(i <- Range(0, seqLen)) {
      val wi = sequence(i)
      val windowStart = 0.max(i - windowSize)
      val windowEnd = seqLen.min(i + windowSize + 1)
      for (j <- windowStart until windowEnd) {
        if (j != i) {
          val wj = sequence(j)
          couples.append((wi, wj))
          labels.append(1)
        }
      }
    }

    if (negativeSamples > 0.0) {
      val numNegSamples = (labels.length * negativeSamples).toInt
      val words = Random.shuffle(couples.map(_._1).toList).toArray
      val random = Random
      for (i <- Range(0, numNegSamples)) {
        val wi = words(i % words.length)
        val wj = random.nextInt(vocabularySize)
        couples.append((wi, wj))
        labels.append(0)
      }
    }

    (couples.toArray, labels.toArray)
  }

}
