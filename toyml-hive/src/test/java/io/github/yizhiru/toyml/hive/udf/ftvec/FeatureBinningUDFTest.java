package io.github.yizhiru.toyml.hive.udf.ftvec;

import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.*;

public class FeatureBinningUDFTest {

    @Test
    public void evaluate() {
        FeatureBinningUDF udf = new FeatureBinningUDF();
        List<Double> percentiles = Arrays.asList(3d, 10d, 20d, 40d, 70d, 140d, 220d, 400d, 800d);
        assertEquals(0, udf.evaluate(1d, percentiles), 0);
        assertEquals(0, udf.evaluate(3d, percentiles), 0);
        assertEquals(1, udf.evaluate(4d, percentiles), 0);
        assertEquals(8, udf.evaluate(401d, percentiles), 0);
        assertEquals(8, udf.evaluate(800d, percentiles), 0);
        assertEquals(9, udf.evaluate(801d, percentiles), 0);
    }
}