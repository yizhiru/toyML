package io.github.yizhiru.toyml.hive.udf.ftvec;

import com.google.common.hash.HashFunction;
import com.google.common.hash.Hashing;
import org.apache.hadoop.hive.ql.exec.UDF;

import java.nio.charset.StandardCharsets;

/**
 * 离散特征Hash，相当于tf.feature_column.categorical_column_with_hash_bucket 函数
 * 特征哈希值属于区间 [1, bucketSize-1]，其中，0值特定为padding用
 */
public final class CategoricalFeatureHashUDF extends UDF {

    private static final HashFunction HASH_FUNCTION = Hashing.murmur3_32();

    /**
     * 特征hash公式：Hash(input_feature_string) % (bucket_size-1) + 1
     *
     * @param value      特征值
     * @param bucketSize 分桶数
     * @return 特征哈希值
     */
    public int evaluate(String value, int bucketSize) {
        if (bucketSize <= 1) {
            return 0;
        }
        if (value == null) {
            value = "";
        }

        int bit = HASH_FUNCTION.hashString(value, StandardCharsets.UTF_8).asInt();
        return Math.floorMod(bit, bucketSize - 1) + 1;
    }
}
