package io.github.yizhiru.toyml.util

import org.scalatest.FunSuite

class SequenceUtilsTest extends FunSuite {

  test("testSkipGrams") {
    val sequence = Array(1, 2, 3)
    val (pairs, labels) = SequenceUtils.skipGram(sequence,
      vocabularySize = 4,
      windowSize = 2,
      negativeSamples = 1.5)
    pairs.zip(labels).foreach{ t =>
      printf("%s\t%d\n", t._1, t._2)
    }
  }

}
