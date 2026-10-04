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

import io.github.jbellis.jvector.disk.ByteBufferIndexWriter;
import io.github.jbellis.jvector.disk.ByteBufferReader;
import io.github.jbellis.jvector.graph.ListRandomAccessVectorValues;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FusedFeature;
import io.github.jbellis.jvector.graph.disk.feature.FusedPQ;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.Int2ObjectHashMap;
import org.junit.Test;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import static org.junit.Assert.*;

/** Deterministic coverage independent of the serving benchmark datasets. */
public class TestSearchCodebookLayout {
    private static final VectorTypeSupport TYPES = VectorizationProvider.getInstance().getVectorTypeSupport();
    private static final int CURRENT = OnDiskGraphIndex.CURRENT_VERSION;

    private static ProductQuantization compressor(int dimension, int subspaces, int clusters, boolean centered, float threshold) {
        Random random = new Random(71L * dimension + clusters);
        int[][] layout = ProductQuantization.getSubvectorSizesAndOffsets(dimension, subspaces);
        VectorFloat<?>[] books = new VectorFloat<?>[subspaces];
        for (int m = 0; m < subspaces; m++) {
            books[m] = TYPES.createFloatVector(layout[m][0] * clusters);
            for (int c = 0; c < clusters; c++) for (int d = 0; d < layout[m][0]; d++) {
                // Different coordinate variances and correlated centroids; no normalization.
                books[m].set(c * layout[m][0] + d, (float) (random.nextGaussian() * (d + 1) + Math.sin(c)));
            }
        }
        var center = centered ? TYPES.createFloatVector(dimension) : null;
        if (center != null) for (int d = 0; d < dimension; d++) center.set(d, (float) random.nextGaussian());
        return new ProductQuantization(books, clusters, layout, center, threshold);
    }

    private static VectorFloat<?> query(ProductQuantization pq, int index) {
        var result = TYPES.createFloatVector(pq.getOriginalDimension());
        float scale = new float[] {0.001f, 1f, 1000f}[index % 3];
        for (int d = 0; d < result.length(); d++) {
            float value = (float) Math.sin(d * 0.7 + index) * scale;
            result.set(d, value + (pq.globalCentroid == null ? 0 : pq.globalCentroid.get(d)));
        }
        return result;
    }

    private static ByteSequence<?>[] codes(ProductQuantization pq) {
        ByteSequence<?>[] result = new ByteSequence<?>[16];
        Random random = new Random(831);
        for (int i = 0; i < result.length; i++) {
            result[i] = TYPES.createByteSequence(pq.getSubspaceCount());
            for (int m = 0; m < pq.getSubspaceCount(); m++) {
                int c = i == 0 ? 0 : i == 1 ? pq.getClusterCount() - 1 : random.nextInt(pq.getClusterCount());
                result[i].set(m, (byte) c);
            }
        }
        return result;
    }

    private static byte[] bytes(ProductQuantization pq, int version) throws Exception {
        var out = ByteBufferIndexWriter.create(pq.compressorSize() + 32, false);
        pq.write(out, version);
        var data = out.getWrittenData();
        byte[] result = new byte[data.remaining()]; data.get(result);
        return result;
    }

    private static byte[] bytes(PQVectors cv, int version) throws Exception {
        var out = ByteBufferIndexWriter.create(cv.pq.compressorSize() + cv.count() * cv.pq.getSubspaceCount() + 64, false);
        cv.write(out, version);
        var data = out.getWrittenData();
        byte[] result = new byte[data.remaining()]; data.get(result);
        return result;
    }

    private static ProductQuantization search(ProductQuantization pq) throws Exception {
        return ProductQuantization.loadForSearch(new ByteBufferReader(ByteBuffer.wrap(bytes(pq, CURRENT))), VectorSimilarityFunction.DOT_PRODUCT);
    }

    @Test public void unchangedBytesAndSingleRepresentation() throws Exception {
        for (boolean centered : new boolean[] {false, true}) {
            var original = compressor(385, 96, 256, centered, KMeansPlusPlusClusterer.UNWEIGHTED);
            for (int version : new int[] {2, 3, CURRENT}) {
                byte[] data = bytes(original, version);
                var reader = new ByteBufferReader(ByteBuffer.wrap(data));
                var loaded = ProductQuantization.loadForSearch(reader, VectorSimilarityFunction.COSINE);
                assertEquals(data.length, reader.getPosition());
                boolean eligible = VectorUtil.supportsTransposedPQ(original.subvectorSizesAndOffsets, original.getClusterCount());
                assertEquals(eligible, loaded.isTransposed());
                assertEquals(eligible, loaded.codebooks == null);
                assertEquals(!eligible, loaded.transposedCodebooks == null);
                if (eligible) assertEquals(385 * 256, loaded.transposedCodebooks.length());
                assertEquals(original, loaded); assertEquals(loaded, original);
                assertEquals(original.hashCode(), loaded.hashCode());
                assertArrayEquals(data, bytes(loaded, version));
                assertEquals(original.compressorSize(), loaded.compressorSize());
                var cv = original.createCompressedVectors(codes(original));
                byte[] vectorBytes = bytes(cv, version);
                var vectorReader = new ByteBufferReader(ByteBuffer.wrap(vectorBytes));
                var loadedVectors = PQVectors.loadForSearch(vectorReader, 0, VectorSimilarityFunction.DOT_PRODUCT);
                assertEquals(vectorBytes.length, vectorReader.getPosition());
                assertEquals(cv, loadedVectors);
                assertArrayEquals(vectorBytes, bytes(loadedVectors, version));
                assertEquals(cv.count(), loadedVectors.count());
            }
        }
    }

    @Test public void encodingCompressorDetachesWithoutChangingSearchOrBytes() throws Exception {
        for (boolean centered : new boolean[] {false, true}) {
            for (float threshold : new float[] {KMeansPlusPlusClusterer.UNWEIGHTED, 0.2f}) {
                var original = compressor(23, 5, 256, centered, threshold);
                var prepared = search(original);
                boolean transposed = prepared.isTransposed();
                var encoder = prepared.forEncoding();
                assertFalse(encoder.isTransposed());
                assertEquals(transposed, prepared.isTransposed());
                assertSame(original, original.forEncoding());
                assertSame(encoder, encoder.forEncoding());
                if (transposed) assertNotSame(prepared, encoder);
                else assertSame(prepared, encoder);
                assertEquals(original, encoder);
                assertArrayEquals(bytes(original, CURRENT), bytes(encoder, CURRENT));
                for (int i = 0; i < 12; i++) assertEquals(original.encode(query(original, i)), encoder.encode(query(original, i)));
                var values = new ListRandomAccessVectorValues(Arrays.asList(query(original,0), null, query(original,2)), 23);
                assertEquals(original.encodeAll(values), encoder.encodeAll(values));
                verify(original, prepared, 1);
            }
        }
    }

    @Test public void cosineMagnitudesPreserveProviderArithmeticBitForBit() throws Exception {
        for (int width : new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 16}) {
            var original = compressor(width * 7, 7, 256, false, KMeansPlusPlusClusterer.UNWEIGHTED);
            var expected = TYPES.createFloatVector(7 * 256);
            for (int m = 0; m < 7; m++)
                VectorUtil.calculatePartialSelfMagnitudes(original.codebooks[m], m, width, 256, expected);
            for (var pq : new ProductQuantization[] {original, search(original)}) {
                var actual = pq.createPartialSquaredMagnitudes();
                for (int i = 0; i < expected.length(); i++)
                    assertEquals(Float.floatToIntBits(expected.get(i)), Float.floatToIntBits(actual.get(i)));
            }
        }
    }

    @Test public void fallbackForUnmeasuredShapesAndEuclidean() throws Exception {
        for (int[] shape : new int[][] {{33, 33}, {264, 33}, {297, 33}, {528, 33}, {385, 48}})
            for (int clusters : new int[] {3, 17, 256}) {
                var pq = compressor(shape[0], shape[1], clusters, false, KMeansPlusPlusClusterer.UNWEIGHTED);
                var loaded = search(pq);
                boolean eligible = VectorUtil.supportsTransposedPQ(pq.subvectorSizesAndOffsets, clusters);
                assertEquals(eligible, loaded.isTransposed());
                assertEquals(eligible, loaded.codebooks == null);
                assertEquals(!eligible, loaded.transposedCodebooks == null);
                var euclidean = ProductQuantization.loadForSearch(new ByteBufferReader(ByteBuffer.wrap(bytes(pq, CURRENT))), VectorSimilarityFunction.EUCLIDEAN);
                assertFalse(euclidean.isTransposed());
                assertEquals(pq, euclidean);
            }
    }

    private static float independent(ProductQuantization pq, ByteSequence<?> code, VectorFloat<?> query, VectorSimilarityFunction similarity) {
        double dot = 0, nq = 0, nv = 0, distance = 0;
        for (int m = 0; m < pq.M; m++) for (int d = 0; d < pq.subvectorSizesAndOffsets[m][0]; d++) {
            int position = pq.subvectorSizesAndOffsets[m][1] + d;
            float q = query.get(position) - (pq.globalCentroid == null ? 0 : pq.globalCentroid.get(position));
            float v = pq.codebooks[m].get(Byte.toUnsignedInt(code.get(m)) * pq.subvectorSizesAndOffsets[m][0] + d);
            dot += (double) q * v; nq += (double) q * q; nv += (double) v * v;
            distance += ((double) q - v) * ((double) q - v);
        }
        switch (similarity) {
            case DOT_PRODUCT: return (float) ((1 + dot) / 2);
            case COSINE: return (float) ((1 + dot / Math.sqrt(nq * nv)) / 2);
            case EUCLIDEAN: return (float) (1 / (1 + distance));
            default: throw new AssertionError();
        }
    }

    private static void close(float expected, float actual) {
        assertTrue("nonfinite score " + actual, Float.isFinite(actual));
        assertEquals(expected, actual, Math.max(0.000003f, Math.abs(expected) * 0.000006f));
    }

    private static void closeScore(ProductQuantization pq, ByteSequence<?> code, VectorFloat<?> query,
                                   VectorSimilarityFunction similarity, float expected, float actual) {
        if (similarity != VectorSimilarityFunction.DOT_PRODUCT) { close(expected,actual); return; }
        // Dot-product cancellation can make relative error in the final sum misleading.
        // Bound rounding by the magnitudes of the terms actually accumulated.
        double absoluteProducts = 0;
        for(int m=0;m<pq.M;m++) for(int d=0;d<pq.subvectorSizesAndOffsets[m][0];d++) {
            int position=pq.subvectorSizesAndOffsets[m][1]+d;
            float q=query.get(position)-(pq.globalCentroid==null?0:pq.globalCentroid.get(position));
            float v=pq.codebooks[m].get(Byte.toUnsignedInt(code.get(m))*pq.subvectorSizesAndOffsets[m][0]+d);
            absoluteProducts+=Math.abs((double)q*v);
        }
        assertTrue(Float.isFinite(actual));
        assertEquals(expected,actual,Math.max(0.000003,absoluteProducts*0.0000005));
    }

    private static void verify(ProductQuantization original, ProductQuantization prepared, int qi) {
        var cs = codes(original);
        var a = original.createCompressedVectors(cs);
        var b = prepared.createCompressedVectors(cs);
        var q = query(original, qi);
        for (var similarity : VectorSimilarityFunction.values()) {
            var baseline = a.precomputedScoreFunctionFor(q, similarity);
            float[] expected = new float[cs.length];
            for (int i = 0; i < cs.length; i++) expected[i] = baseline.similarityTo(i);
            var actual = b.precomputedScoreFunctionFor(q, similarity);
            var simple = b.scoreFunctionFor(q, similarity);
            for (int i = 0; i < cs.length; i++) {
                closeScore(original, cs[i], q, similarity, independent(original, cs[i], q, similarity), actual.similarityTo(i));
                closeScore(original, cs[i], q, similarity, expected[i], actual.similarityTo(i));
                closeScore(original, cs[i], q, similarity, expected[i], simple.similarityTo(i));
            }
        }
    }

    @Test public void independentScoresIncludingNonUnitDotProduct() throws Exception {
        for (int[] shape : new int[][] {{33,33}, {66,33}, {99,33}, {132,33}, {165,33}, {198,33},
                {231,33}, {264,33}, {297,33}, {528,33}, {384,96}, {1024,192}, {385,48}})
            for (boolean center : new boolean[] {false,true}) {
                var original = compressor(shape[0], shape[1], 256, center, KMeansPlusPlusClusterer.UNWEIGHTED);
                var prepared = search(original);
                for (int q = 0; q < 9; q++) verify(original, prepared, q);
            }
    }

    @Test public void encodingDecodingRefinementAndDiversity() throws Exception {
        for (float threshold : new float[] {KMeansPlusPlusClusterer.UNWEIGHTED, 0.2f}) {
            var original = compressor(23,8,256,true,threshold);
            var prepared = search(original);
            for (int q = 0; q < 9; q++) {
                var vector = query(original,q);
                assertEquals(original.encode(vector), prepared.encode(vector));
                assertEquals(original.reconstructionError(vector), prepared.reconstructionError(vector), 1e-10);
                var code = codes(original)[q];
                var a = TYPES.createFloatVector(23); var b = TYPES.createFloatVector(23);
                original.decode(code,a); prepared.decode(code,b); assertEquals(a,b);
            }
            assertEquals(original.getOrComputeCentroid(), prepared.getOrComputeCentroid());
            var a = original.createCompressedVectors(codes(original));
            var b = prepared.createCompressedVectors(codes(original));
            for (var similarity : VectorSimilarityFunction.values()) for (int n = 0; n < a.count(); n++) {
                var first = a.diversityFunctionFor(n,similarity); var second = b.diversityFunctionFor(n,similarity);
                for (int i=0;i<a.count();i++) close(first.similarityTo(i),second.similarityTo(i));
            }
            for (var similarity : new VectorSimilarityFunction[] {VectorSimilarityFunction.DOT_PRODUCT,VectorSimilarityFunction.EUCLIDEAN}) {
                assertEquals(original.createCodebookPartialSums(similarity),prepared.createCodebookPartialSums(similarity));
            }
            var training = new ArrayList<VectorFloat<?>>();
            // Give every initial centroid assigned points, so empty-cluster random
            // reseeding cannot make two otherwise identical refinements differ.
            for (int c = 0; c < 256; c++) for (int copy = 0; copy < 4; copy++) {
                var code = TYPES.createByteSequence(original.M);
                for (int m=0;m<original.M;m++) code.set(m,(byte)c);
                var point = TYPES.createFloatVector(23); original.decode(code,point);
                for(int d=0;d<23;d++) point.set(d,point.get(d)+copy*0.00001f);
                training.add(point);
            }
            var values = new ListRandomAccessVectorValues(training,23);
            var refinedOriginal = original.refine(values,1,threshold,ForkJoinPool.commonPool(),ForkJoinPool.commonPool());
            var refinedPrepared = prepared.refine(values,1,threshold,ForkJoinPool.commonPool(),ForkJoinPool.commonPool());
            assertEquals(refinedOriginal,refinedPrepared);
            assertFalse(original.equals(refinedPrepared));
            var missing = new ListRandomAccessVectorValues(Arrays.asList(query(original,0),null,query(original,2)),23);
            assertEquals(original.encodeAll(missing),prepared.encodeAll(missing));
        }
    }

    @Test public void fusedNeighborAndHierarchyScores() throws Exception {
        for (boolean centered : new boolean[] {false,true}) {
            var original = compressor(39,13,256,centered,KMeansPlusPlusClusterer.UNWEIGHTED);
            var prepared = search(original);
            var cs = codes(original);
            var feature = new FusedPQ(cs.length,prepared);
            var packed = feature.new PackedNeighbors(null) {
                @Override public void readInto(int node,ByteSequence<?> destination) {
                    for (int i=0;i<cs.length;i++) for(int m=0;m<prepared.M;m++) destination.set(i*prepared.M+m,cs[i].get(m));
                }
            };
            var hierarchy = new Int2ObjectHashMap<FusedFeature.InlineSource>();
            for (int i=0;i<cs.length;i++) hierarchy.put(i,new FusedPQ.FusedPQInlineSource(cs[i]));
            for (var similarity : VectorSimilarityFunction.values()) {
                var q=query(original,1);
                var baseline=original.createCompressedVectors(cs).precomputedScoreFunctionFor(q,similarity);
                var fused=FusedPQDecoder.newDecoder(packed,prepared,hierarchy,q,TYPES.createByteSequence(cs.length*prepared.M),
                        TYPES.createFloatVector(cs.length),similarity,node -> 0);
                fused.enableSimilarityToNeighbors(0);
                for(int i=0;i<cs.length;i++) {
                    close(baseline.similarityTo(i),fused.similarityTo(i));
                    close(baseline.similarityTo(i),fused.similarityToNeighbor(0,i));
                }
            }
        }
    }

    @Test public void directKernelSupportsIrregularSlicesAndScalarTails() {
        int[] widths={1,2,3,4,5,6,7,8};
        int[] offsets={0,4,10,17,25,34,44,55};
        var q=TYPES.createFloatVector(70);
        for(int d=0;d<70;d++) q.set(d,(float)Math.sin(d));
        for(int clusters:new int[] {3,17,256}) {
            var flat=TYPES.createFloatVector(36*clusters);
            for(int i=0;i<flat.length();i++) flat.set(i,(float)Math.cos(i*0.31));
            var output=TYPES.createFloatVector(widths.length*clusters);
            VectorUtil.calculateTransposedPQDotProducts(flat,widths,offsets,clusters,q,output);
            int base=0;
            for(int m=0;m<widths.length;m++) {
                for(int c=0;c<clusters;c++) {
                    double expected=0;
                    for(int d=0;d<widths[m];d++) expected+=(double)flat.get(base+d*clusters+c)*q.get(offsets[m]+d);
                    close((float)expected,output.get(m*clusters+c));
                }
                base+=widths[m]*clusters;
            }
        }
    }

    @Test public void invalidNativeInputsAreRejectedBeforeDispatch() {
        var coordinates = TYPES.createFloatVector(256);
        var query = TYPES.createFloatVector(8);
        var output = TYPES.createFloatVector(256);
        int[][] widths = {{1}, {0}, {-1}, {9}, {1}, {1}, {1}, {1}, {Integer.MAX_VALUE}};
        int[][] offsets = {{0,1}, {0}, {0}, {0}, {-1}, {8}, {Integer.MAX_VALUE}, {8}, {0}};
        for (int i = 0; i < widths.length; i++) {
            try {
                VectorUtil.calculateTransposedPQDotProducts(coordinates, widths[i], offsets[i], 256, query, output);
                fail("Accepted invalid query slice " + i);
            } catch (IllegalArgumentException expected) {}
        }
        for (int clusters : new int[] {0, -1, Integer.MAX_VALUE}) {
            try {
                VectorUtil.calculateTransposedPQDotProducts(coordinates, new int[] {1}, new int[] {0}, clusters, query, output);
                fail("Accepted invalid cluster/buffer size");
            } catch (IllegalArgumentException expected) {}
        }
        try {
            VectorUtil.calculateTransposedPQDotProducts(coordinates, new int[] {1}, new int[] {0}, 256,
                query, TYPES.createFloatVector(255));
            fail("Accepted undersized output");
        } catch (IllegalArgumentException expected) {}
        try {
            VectorUtil.calculateTransposedPQDotProducts(TYPES.createFloatVector(255), new int[] {1}, new int[] {0},
                256, query, output);
            fail("Accepted undersized coordinates");
        } catch (IllegalArgumentException expected) {}
    }

    @Test public void concurrentQueryScratchIsIndependent() throws Exception {
        var original=compressor(384,96,256,true,KMeansPlusPlusClusterer.UNWEIGHTED);
        var prepared=search(original);
        var pool=Executors.newFixedThreadPool(4);
        try {
            var futures=new ArrayList<java.util.concurrent.Future<?>>();
            for(int thread=0;thread<4;thread++) {
                final int index=thread;
                futures.add(pool.submit(() -> { for(int i=0;i<12;i++) verify(original,prepared,index+i); }));
            }
            for(var f:futures) f.get();
        } finally { pool.shutdownNow(); }
    }
}
