/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.github.jbellis.jvector.quantization;

import io.github.jbellis.jvector.graph.similarity.ScoreFunction;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;

/**
 * Performs similarity comparisons with compressed vectors without decoding them
 */
abstract class PQDecoder implements ScoreFunction.ApproximateScoreFunction {

    protected final PQVectors cv;

    protected PQDecoder(PQVectors cv) {
        this.cv = cv;
    }

    protected static abstract class CachingDecoder extends PQDecoder {
        protected final VectorFloat<?> partialSums;

        protected CachingDecoder(PQVectors cv, VectorFloat<?> query, VectorSimilarityFunction vsf) {
            super(cv);
            var pq = this.cv.pq;
            partialSums = cv.reusablePartialSums();

            VectorFloat<?> center = pq.globalCentroid;
            var centeredQuery = center == null ? query : VectorUtil.sub(query, center);
            pq.calculatePartialSums(centeredQuery, vsf, partialSums);
        }

        protected float decodedSimilarity(ByteSequence<?> encoded, int offset, int length) {
            return VectorUtil.assembleAndSum(partialSums, cv.pq.getClusterCount(), encoded, offset, length);
        }
    }

    static class DotProductDecoder extends CachingDecoder {
        public DotProductDecoder(PQVectors cv, VectorFloat<?> query) {
            super(cv, query, VectorSimilarityFunction.DOT_PRODUCT);
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedSimilarity(cv.getChunk(node2), cv.getOffsetInChunk(node2), cv.pq.getSubspaceCount())) / 2;
        }
    }

    static class EuclideanDecoder extends CachingDecoder {
        public EuclideanDecoder(PQVectors cv, VectorFloat<?> query) {
            super(cv, query, VectorSimilarityFunction.EUCLIDEAN);
        }

        @Override
        public float similarityTo(int node2) {
            return 1 / (1 + decodedSimilarity(cv.getChunk(node2), cv.getOffsetInChunk(node2), cv.pq.getSubspaceCount()));
        }
    }

    static class CosineDecoder extends PQDecoder {
        protected final VectorFloat<?> partialSums;
        protected final VectorFloat<?> aMagnitude;
        protected final float bMagnitude;

        public CosineDecoder(PQVectors cv, VectorFloat<?> query) {
            super(cv);
            var pq = this.cv.pq;

            // this part is not query-dependent, so we can cache it
            aMagnitude = cv.partialSquaredMagnitudes().updateAndGet(current -> {
                if (current != null) {
                    return current;
                }

                return pq.createPartialSquaredMagnitudes();
            });


            // Compute and cache partial sums and magnitudes for query vector
            partialSums = cv.reusablePartialSums();

            VectorFloat<?> center = pq.globalCentroid;
            VectorFloat<?> centeredQuery = center == null ? query : VectorUtil.sub(query, center);

            pq.calculatePartialSums(centeredQuery, VectorSimilarityFunction.DOT_PRODUCT, partialSums);

            this.bMagnitude = VectorUtil.dotProduct(centeredQuery, centeredQuery);
        }

        @Override
        public float similarityTo(int node2) {
            return (1 + decodedCosine(node2)) / 2;
        }

        protected float decodedCosine(int node2) {

            ByteSequence<?> encoded = cv.getChunk(node2);
            int offset = cv.getOffsetInChunk(node2);

            return VectorUtil.pqDecodedCosineSimilarity(encoded, offset, cv.pq.getSubspaceCount(), cv.pq.getClusterCount(), partialSums, aMagnitude, bMagnitude);
        }
    }
}
