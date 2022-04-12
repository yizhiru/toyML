package io.github.yizhiru.toyml.util

import org.scalatest.FunSuite

class RandomUtilsTest extends FunSuite {

  test("testRandomChoice") {
    val result = Range(0, 1000).map { _ =>
      RandomUtils.randomChoice(Array(0,1,2,3), Array(0.5, 0.1, 0.1, 0.3))
    }
      .groupBy(e => e)
      .map(t => (t._1, t._2.length))
    println(result)
  }

}
