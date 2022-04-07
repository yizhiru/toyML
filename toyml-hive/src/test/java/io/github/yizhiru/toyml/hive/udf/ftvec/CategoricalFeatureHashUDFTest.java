package io.github.yizhiru.toyml.hive.udf.ftvec;

import org.junit.Test;

import java.util.stream.IntStream;

public class CategoricalFeatureHashUDFTest {

    @Test
    public void evaluate() {
        CategoricalFeatureHashUDF udf = new CategoricalFeatureHashUDF();
        for (int i: IntStream.range(0, 20).toArray()) {
            System.out.println(udf.evaluate(String.valueOf(i), 10));
        }

    }

}