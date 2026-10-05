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

import io.github.jbellis.jvector.annotations.VisibleForTesting;
import io.github.jbellis.jvector.disk.IndexWriter;
import io.github.jbellis.jvector.disk.RandomAccessReader;
import io.github.jbellis.jvector.graph.RandomAccessVectorValues;
import io.github.jbellis.jvector.graph.VectorAccess;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.util.Accountable;
import io.github.jbellis.jvector.util.PhysicalCoreExecutor;
import io.github.jbellis.jvector.vector.VectorSimilarityFunction;
import io.github.jbellis.jvector.vector.VectorUtil;
import io.github.jbellis.jvector.vector.VectorizationProvider;
import io.github.jbellis.jvector.vector.types.ByteSequence;
import io.github.jbellis.jvector.vector.types.VectorFloat;
import io.github.jbellis.jvector.vector.types.VectorTypeSupport;
import org.agrona.collections.IntHashSet;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static io.github.jbellis.jvector.quantization.KMeansPlusPlusClusterer.UNWEIGHTED;
import static io.github.jbellis.jvector.util.MathUtil.square;
import static io.github.jbellis.jvector.vector.VectorUtil.dotProduct;
import static io.github.jbellis.jvector.vector.VectorUtil.sub;
import static java.lang.Math.sqrt;

/**
 * Product Quantization for float vectors.  Supports arbitrary source and target dimensionality;
 * in particular, the source does not need to be evenly divisible by the target.
 */
public class ProductQuantization implements VectorCompressor<ByteSequence<?>>, Accountable {
    private static final int MAGIC = 0x75EC4012; // JVECTOR, with some imagination

    protected static final Logger LOG = Logger.getLogger(ProductQuantization.class.getName());

    private static final VectorTypeSupport vectorTypeSupport = VectorizationProvider.getInstance().getVectorTypeSupport();
    static final int DEFAULT_CLUSTERS = 256; // number of clusters per subspace = one byte's worth
    public static final int K_MEANS_ITERATIONS = 6;
    public static final int MAX_PQ_TRAINING_SET_SIZE = 128000;

    // Exactly one coordinate representation is retained. Training uses the
    // original per-codebook layout; eligible search loads use one flat layout.
    final VectorFloat<?> transposedCodebooks;
    private final int[] subvectorWidths;
    private final int[] subvectorOffsets;
    private final ThreadLocal<VectorFloat<?>[]> centroidScratch;
    final VectorFloat<?>[] codebooks; // centroid-major coordinates; null for transposed search loads
    final int M; // number of subspaces
    private final int clusterCount; // centroids per subspace
    final int originalDimension;
    final VectorFloat<?> globalCentroid;
    final int[][] subvectorSizesAndOffsets;
    final float anisotropicThreshold; // parallel cost multiplier
    private final float[][] centroidNormsSquared; // precomputed norms of the centroids, for encoding
    private final ThreadLocal<VectorFloat<?>> partialSums; // for dot product, euclidean, and cosine partials
    private final AtomicReference<VectorFloat<?>> partialSquaredMagnitudes; // standalone cosine partials
    private final AtomicReference<VectorFloat<?>> fusedPartialSquaredMagnitudes; // fused cosine uses dot-product arithmetic

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter) {
        return compute(ravv, M, clusterCount, globallyCenter, UNWEIGHTED, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    public static ProductQuantization compute(RandomAccessVectorValues ravv, int M, int clusterCount, boolean globallyCenter, float anisotropicThreshold) {
        return compute(ravv, M, clusterCount, globallyCenter, anisotropicThreshold, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Initializes the codebooks by clustering the input data using Product Quantization.
     *
     * @param ravv the vectors to quantize
     * @param M number of subspaces
     * @param clusterCount number of clusters per subspace
     * @param globallyCenter whether to center the vectors globally before quantization
     *                       (not recommended when using the quantization for dot product)
     * @param anisotropicThreshold the threshold of relevance for anisotropic angular distance shaping, giving
     *        higher priority to parallel error.  Anisotropic shaping requires that your dataset be normalized
     *        to unit length.  Use a threshold of UNWEIGHTED for isotropic distance
     *        (i.e. normal, unweighted L2 distance).
     * @param simdExecutor     ForkJoinPool instance for SIMD operations, best is to use a pool with the size of
     *                         the number of physical cores.
     * @param parallelExecutor ForkJoinPool instance for parallel stream operations
     */
    public static ProductQuantization compute(RandomAccessVectorValues ravv,
                                              int M,
                                              int clusterCount,
                                              boolean globallyCenter,
                                              float anisotropicThreshold,
                                              ForkJoinPool simdExecutor,
                                              ForkJoinPool parallelExecutor)
    {
        checkClusterCount(clusterCount);

        if (ravv.size() < clusterCount) {
            throw new IllegalArgumentException("Cannot train PQ with %d clusters on %d points, supply more training vectors or lower cluster count.");
        }
        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectors = extractTrainingVectors(ravv, parallelExecutor);

        // subtract the centroid from each training vector
        VectorFloat<?> globalCentroid;
        if (globallyCenter) {
            globalCentroid = KMeansPlusPlusClusterer.centroidOf(vectors);
            // subtract the centroid from each vector
            List<VectorFloat<?>> finalVectors = vectors;
            vectors = simdExecutor.submit(() -> finalVectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.<VectorFloat<?>>toList())).join();
        } else {
            globalCentroid = null;
        }

        // derive the codebooks
        var codebooks = createCodebooks(vectors, subvectorSizesAndOffsets, clusterCount, anisotropicThreshold, simdExecutor);
        return new ProductQuantization(codebooks, clusterCount, subvectorSizesAndOffsets, globalCentroid, anisotropicThreshold);
    }

    static List<VectorFloat<?>> extractTrainingVectors(RandomAccessVectorValues ravv, ForkJoinPool parallelExecutor) {
        final IntStream ordinalStream;

        if (ravv.size() <= MAX_PQ_TRAINING_SET_SIZE) {
            ordinalStream = IntStream.range(0, ravv.size());
        } else {
            // Uses Floyd’s sampling algorithm to select MAX_PQ_TRAINING_SET_SIZE random ordinals from 0 to ravv.size()
            // while only iterating MAX_PQ_TRAINING_SET_SIZE times.
            SplittableRandom rng = new SplittableRandom(1);
            IntHashSet ordinals = new IntHashSet(MAX_PQ_TRAINING_SET_SIZE);
            // j runs from (ravv.size() - MAX_PQ_TRAINING_SET_SIZE) to ravv.size() (exclusive)
            for (int j = ravv.size() - MAX_PQ_TRAINING_SET_SIZE; j < ravv.size(); j++) {
                int t = rng.nextInt(j + 1); // uniform in [0, j]
                if (ordinals.contains(t)) {
                    ordinals.add(j);
                } else {
                    ordinals.add(t);
                }
            }
            int[] ordinalArray = new int[ordinals.size()];
            IntHashSet.IntIterator it = ordinals.iterator();
            for (int i = 0; i < ordinals.size(); i++) {
                assert it.hasNext();
                ordinalArray[i] = it.next();
            }
            assert !it.hasNext();
            ordinalStream = IntStream.of(ordinalArray);
        }

        return VectorAccess.copySelected(ravv, ordinalStream.toArray(), parallelExecutor);
    }

    /**
     * Create a new PQ by fine-tuning this one with the data in `ravv`
     */
    public ProductQuantization refine(RandomAccessVectorValues ravv) {
        return refine(ravv, 1, UNWEIGHTED, PhysicalCoreExecutor.pool(), ForkJoinPool.commonPool());
    }

    /**
     * Create a new PQ by fine-tuning this one with the data in `ravv`
     *
     * @param lloydsRounds number of Lloyd's iterations to run against
     *                     the new data.  Suggested values are 1 or 2.
     */
    public ProductQuantization refine(RandomAccessVectorValues ravv,
                                      int lloydsRounds,
                                      float anisotropicThreshold,
                                      ForkJoinPool simdExecutor,
                                      ForkJoinPool parallelExecutor)
    {
        if (lloydsRounds < 0) {
            throw new IllegalArgumentException("lloydsRounds must be non-negative");
        }

        var subvectorSizesAndOffsets = getSubvectorSizesAndOffsets(ravv.dimension(), M);
        var vectorsMutable = extractTrainingVectors(ravv, parallelExecutor);
        if (globalCentroid != null) {
            var vectors = vectorsMutable;
            vectorsMutable = simdExecutor.submit(() -> vectors.stream().parallel().map(v -> VectorUtil.sub(v, globalCentroid)).collect(Collectors.<VectorFloat<?>>toList())).join();
        }
        var vectors = vectorsMutable; // "effectively final" to make the closure happy

        Callable<VectorFloat<?>[]> callable = () -> IntStream.range(0, M).parallel().mapToObj(m -> {
            VectorFloat<?>[] subvectors = extractSubvectors(vectors, m, subvectorSizesAndOffsets);
            var clusterer = new KMeansPlusPlusClusterer(subvectors, centroidMajorCodebook(m), anisotropicThreshold);
            return clusterer.cluster(anisotropicThreshold == UNWEIGHTED ? lloydsRounds : 0,
                                     anisotropicThreshold == UNWEIGHTED ? 0 : lloydsRounds);
        }).toArray(VectorFloat<?>[]::new);
        var refinedCodebooks = simdExecutor.submit(callable).join();

        return new ProductQuantization(refinedCodebooks, clusterCount, subvectorSizesAndOffsets, globalCentroid, anisotropicThreshold);
    }

    ProductQuantization(VectorFloat<?>[] codebooks, int clusterCount, int[][] subvectorSizesAndOffsets, VectorFloat<?> globalCentroid, float anisotropicThreshold) {
        this(codebooks, null, clusterCount, subvectorSizesAndOffsets, globalCentroid, anisotropicThreshold);
    }

    private ProductQuantization(VectorFloat<?>[] codebooks, VectorFloat<?> transposedCodebooks,
                                int clusterCount, int[][] subvectorSizesAndOffsets,
                                VectorFloat<?> globalCentroid, float anisotropicThreshold) {
        checkClusterCount(clusterCount);
        if ((codebooks == null) == (transposedCodebooks == null)) {
            throw new IllegalArgumentException("Exactly one codebook representation is required");
        }
        this.transposedCodebooks = transposedCodebooks;
        subvectorWidths = Arrays.stream(subvectorSizesAndOffsets).mapToInt(s -> s[0]).toArray();
        subvectorOffsets = Arrays.stream(subvectorSizesAndOffsets).mapToInt(s -> s[1]).toArray();
        int maxWidth = Arrays.stream(subvectorWidths).max().orElseThrow();
        centroidScratch = transposedCodebooks == null ? null : ThreadLocal.withInitial(() -> new VectorFloat<?>[] {
            vectorTypeSupport.createFloatVector(maxWidth), vectorTypeSupport.createFloatVector(maxWidth)
        });

        this.codebooks = codebooks;
        this.globalCentroid = globalCentroid;
        this.M = subvectorSizesAndOffsets.length;
        this.clusterCount = clusterCount;
        this.subvectorSizesAndOffsets = subvectorSizesAndOffsets;
        this.originalDimension = Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum();
        if (globalCentroid != null && globalCentroid.length() != originalDimension) {
            var msg = String.format("Global centroid length %d does not match vector dimensionality %d", globalCentroid.length(), originalDimension);
            throw new IllegalArgumentException(msg);
        }
        this.anisotropicThreshold = anisotropicThreshold;
        this.partialSums = ThreadLocal.withInitial(() -> vectorTypeSupport.createFloatVector(getSubspaceCount() * getClusterCount()));
        this.partialSquaredMagnitudes = new AtomicReference<>(null);
        this.fusedPartialSquaredMagnitudes = new AtomicReference<>(null);


        centroidNormsSquared = new float[M][clusterCount];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < clusterCount; j++) {
                centroidNormsSquared[i][j] = centroidDot(i, j, j);
            }
        }
    }

    @Override
    public ImmutablePQVectors createCompressedVectors(Object[] compressedVectors) {
        return new ImmutablePQVectors(this, (ByteSequence<?>[]) compressedVectors, compressedVectors.length, 1);
    }

    /**
     * Encodes the given vectors in parallel using the PQ codebooks. If a vector is missing (null), it will be encoded
     * as a zero vector.
     */
    @Override
    public PQVectors encodeAll(RandomAccessVectorValues ravv, ForkJoinPool simdExecutor) {
        return PQVectors.encodeAndBuild(this, ravv.size(), ravv, simdExecutor);
    }

    /**
     * Encodes the input vector using the PQ codebooks, weighing parallel loss more than orthogonal loss, into
     * the given ByteSequence.
     */
    private void encodeAnisotropic(VectorFloat<?> vector, ByteSequence<?> result) {
        // compute the residuals from each subvector to each corresponding codebook centroid
        Residual[][] residuals = computeResiduals(vector);
        assert residuals.length == M : "Residuals length mismatch " + residuals.length + " != " + M;
        // start with centroids that minimize the residual norms
        initializeToMinResidualNorms(residuals, result);
        // sum the initial parallel residual component
        float parallelResidualComponentSum = 0;
        for (int i = 0; i < result.length(); i++) {
            int centroidIdx = Byte.toUnsignedInt(result.get(i));
            parallelResidualComponentSum += residuals[i][centroidIdx].parallelResidualComponent;
        }

        // SCANN sorts the subspaces by residual norm here (and adds a sorted->original subspace index map),
        // presumably with the intent to help this converge faster, but profiling shows that almost 90% of the
        // cost of this method is computeResiduals + initializeToMinResidualNorms, so we're not going to bother.

        // Optimize until convergence
        int MAX_ITERATIONS = 10; // borrowed from SCANN code without experimenting w/ other values
        for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
            // loop over each subspace
            boolean changed = false;
            for (int i = 0; i < residuals.length; i++) {
                int oldIdx = Byte.toUnsignedInt(result.get(i));
                CoordinateDescentResult cdr = optimizeSingleSubspace(residuals[i], oldIdx, parallelResidualComponentSum);
                if (cdr.newCenterIdx != oldIdx) {
                    parallelResidualComponentSum = cdr.newParallelResidualComponent;
                    result.set(i, (byte) cdr.newCenterIdx);
                    changed = true;
                }
            }
            // Done if nothing changed this iteration
            if (!changed) {
                break;
            }
        }
    }

    private CoordinateDescentResult optimizeSingleSubspace(Residual[] residuals, int oldIdx, float oldParallelResidualSum) {
        // (this is global to all subspaces but it's not worth stashing in a field)
        float pcm = KMeansPlusPlusClusterer.computeParallelCostMultiplier(anisotropicThreshold, originalDimension);

        float oldResidualNormSquared = residuals[oldIdx].residualNormSquared;
        float oldParallelComponent = residuals[oldIdx].parallelResidualComponent;

        float bestCostDelta = 0;
        int bestIndex = oldIdx;
        float bestParallelResidualSum = oldParallelResidualSum;

        // loop over potential new centers
        for (int thisIdx = 0; thisIdx < residuals.length; thisIdx++) {
            if (thisIdx == oldIdx) {
                continue;
            }

            // compute the new parallel residual sum and parallel norm delta
            Residual rs = residuals[thisIdx];
            float thisParallelResidualSum = oldParallelResidualSum - oldParallelComponent + rs.parallelResidualComponent;
            float parallelNormDelta = square(thisParallelResidualSum) - square(oldParallelResidualSum);
            // quit early if new parallel norm is worse than the old
            if (parallelNormDelta > 0) {
                continue;
            }

            // compute the total cost delta
            float residualNormDelta = rs.residualNormSquared - oldResidualNormSquared;
            float perpendicularNormDelta = residualNormDelta - parallelNormDelta;
            float costDelta = pcm * parallelNormDelta + perpendicularNormDelta;

            // save the new center if it's the best so far
            if (costDelta < bestCostDelta) {
                bestCostDelta = costDelta;
                bestIndex = thisIdx;
                bestParallelResidualSum = thisParallelResidualSum;
            }
        }

        return new CoordinateDescentResult(bestIndex, bestParallelResidualSum);
    }

    /**
     * Wraps the two values we want to return from optimizeSingleSubspace
     */
    private static class CoordinateDescentResult {
        final int newCenterIdx;
        final float newParallelResidualComponent;

        CoordinateDescentResult(int newCenterIdx, float newParallelResidualComponent) {
            this.newCenterIdx = newCenterIdx;
            this.newParallelResidualComponent = newParallelResidualComponent;
        }
    }

    /**
     * @return codebook ordinals representing the cluster centroids for each subspace that minimize the residual norm
     */
    private void initializeToMinResidualNorms(Residual[][] residualStats, ByteSequence<?> dest) {
        // for each subspace
        for (int i = 0; i < residualStats.length; i++) {
            int minIndex = -1;
            double minNormSquared = Double.MAX_VALUE;
            // find the centroid with the smallest residual norm in this subspace
            for (int j = 0; j < residualStats[i].length; j++) {
                if (residualStats[i][j].residualNormSquared < minNormSquared) {
                    minNormSquared = residualStats[i][j].residualNormSquared;
                    minIndex = j;
                }
            }
            dest.set(i, (byte) minIndex);
        }
    }

    /**
     * @return the parallel-cost residuals for each subspace and cluster
     */
    private Residual[][] computeResiduals(VectorFloat<?> vector) {
        Residual[][] residuals = new Residual[M][];

        float inverseNorm = (float) (1.0 / sqrt(dotProduct(vector, vector)));
        for (int i = 0; i < M; i++) {
            var x = getSubVector(vector, i, subvectorSizesAndOffsets);
            float xNormSquared = dotProduct(x, x);
            residuals[i] = new Residual[clusterCount];

            for (int j = 0; j < clusterCount; j++) {
                residuals[i][j] = computeResidual(x, i, j, centroidNormsSquared[i][j], xNormSquared, inverseNorm);
            }
        }

        return residuals;
    }

    /**
     * Represents the residual after subtracting a cluster centroid from a [sub]vector.
     */
    private static class Residual {
        final float residualNormSquared;
        final float parallelResidualComponent;

        Residual(float residualNormSquared, float parallelResidualComponent) {
            this.residualNormSquared = residualNormSquared;
            this.parallelResidualComponent = parallelResidualComponent;
        }
    }

    private Residual computeResidual(VectorFloat<?> x, int subspace, int centroid, float cNormSquared, float xNormSquared, float inverseNorm) {
        float cDotX = centroidDot(subspace, centroid, x, 0);
        float residualNormSquared = cNormSquared - 2 * cDotX + xNormSquared;
        float parallelErrorSubtotal = cDotX - xNormSquared;
        float parallelResidualComponent = square(parallelErrorSubtotal) * inverseNorm;
        return new Residual(residualNormSquared, parallelResidualComponent);
    }

    private void encodeUnweighted(VectorFloat<?> vector, ByteSequence<?> dest) {
        for (int m = 0; m < M; m++) {
            dest.set(m, (byte) closestCentroidIndex(vector, m));
        }
    }

    /**
     * Encodes the input vector using the PQ codebooks.
     * @return one byte per subspace
     */
    @Override
    public ByteSequence<?> encode(VectorFloat<?> vector) {
        var result = vectorTypeSupport.createByteSequence(M);
        encodeTo(vector, result);
        return result;
    }

    @Override
    public void encodeTo(VectorFloat<?> vector, ByteSequence<?> dest) {
        if (globalCentroid != null) {
            vector = sub(vector, globalCentroid);
        }

        if (anisotropicThreshold > UNWEIGHTED)
            encodeAnisotropic(vector, dest);
        else
            encodeUnweighted(vector, dest);
    }

    /**
     * Decodes the quantized representation (ByteSequence) to its approximate original vector.
     */
    public void decode(ByteSequence<?> encoded, VectorFloat<?> target) {
        decodeCentered(encoded, target);

        if (globalCentroid != null) {
            // Add back the global centroid to get the approximate original vector.
            VectorUtil.addInPlace(target, globalCentroid);
        }
    }

    /**
     * Decodes the quantized representation (ByteSequence) to its approximate original vector, relative to the global centroid.
     */
    void decodeCentered(ByteSequence<?> encoded, VectorFloat<?> target) {
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(encoded.get(m));
            copyCentroid(m, centroidIndex, target, subvectorSizesAndOffsets[m][1]);
        }
    }

    /**
     * @return how many bytes we are compressing to
     */
    public int getSubspaceCount() {
        return M;
    }

    /**
     * @return number of clusters per subspace
     */
    public int getClusterCount() {
        return clusterCount;
    }

    /** Width of subspace {@code m}'s subvectors. */
    public int getSubvectorSize(int m) {
        return subvectorSizesAndOffsets[m][0];
    }

    /**
     * Subspace {@code m}'s codebook: {@code clusterCount} contiguous subvectors. Read-only use.
     * A transposed search compressor returns a detached centroid-major copy.
     */
    public VectorFloat<?> getCodebookVector(int m) {
        return centroidMajorCodebook(m);
    }

    /** The global centroid subtracted before encoding, or {@code null} when not center-adjusted. */
    public VectorFloat<?> getGlobalCentroid() {
        return globalCentroid;
    }

    static VectorFloat<?>[] createCodebooks(List<VectorFloat<?>> vectors, int[][] subvectorSizeAndOffset, int clusters, float anisotropicThreshold, ForkJoinPool simdExecutor) {
        int M = subvectorSizeAndOffset.length;
        Callable<VectorFloat<?>[]> callable = () -> IntStream.range(0, M).parallel().mapToObj(m -> {
            VectorFloat<?>[] subvectors = extractSubvectors(vectors, m, subvectorSizeAndOffset);
            var clusterer = new KMeansPlusPlusClusterer(subvectors, clusters, anisotropicThreshold);
            return clusterer.cluster(K_MEANS_ITERATIONS, anisotropicThreshold == UNWEIGHTED ? 0 : K_MEANS_ITERATIONS);
        }).toArray(VectorFloat<?>[]::new);
        return simdExecutor.submit(callable).join();
    }

    /**
     * Extract VectorFloat subvectors corresponding to the m'th subspace.
     * This is NOT done in parallel (since the callers are themselves running in parallel).
     */
    private static VectorFloat<?>[] extractSubvectors(List<VectorFloat<?>> vectors, int m, int[][] subvectorSizeAndOffset) {
        return vectors.stream()
                .map(vector -> getSubVector(vector, m, subvectorSizeAndOffset))
                .toArray(VectorFloat<?>[]::new);
    }

    int closestCentroidIndex(VectorFloat<?> subvector, int m) {
        int index = 0;
        float minDist = Float.MAX_VALUE;
        int subvectorOffset = subvectorSizesAndOffsets[m][1];
        // The bulk kernel scans a centroid-major codebook. Sustained encoding with a
        // transposed search compressor should use forEncoding() rather than this path.
        if (!isTransposed()) {
            int bulkNearest = VectorUtil.tryClosestCentroid(subvector, subvectorOffset, codebooks[m], clusterCount, subvectorWidths[m]);
            if (bulkNearest >= 0) {
                return bulkNearest;
            }
        }
        for (int i = 0; i < clusterCount; i++) {
            float dist = centroidDistance(m, i, subvector, subvectorOffset);
            if (dist < minDist) {
                minDist = dist;
                index = i;
            }
        }
        return index;
    }

    /**
     * Extracts the m-th subvector from a single vector.
     */
    static VectorFloat<?> getSubVector(VectorFloat<?> vector, int m, int[][] subvectorSizeAndOffset) {
        VectorFloat<?> subvector = vectorTypeSupport.createFloatVector(subvectorSizeAndOffset[m][0]);
        subvector.copyFrom(vector, subvectorSizeAndOffset[m][1], 0, subvectorSizeAndOffset[m][0]);
        return subvector;
    }

    /**
     * Splits the vector dimension into M subvectors of roughly equal size.
     */
    @VisibleForTesting
    static int[][] getSubvectorSizesAndOffsets(int dimensions, int M) {
        if (M > dimensions) {
            throw new IllegalArgumentException("Number of subspaces must be less than or equal to the vector dimension");
        }
        int[][] sizes = new int[M][];
        int baseSize = dimensions / M;
        int remainder = dimensions % M;
        // distribute the remainder among the subvectors
        int offset = 0;
        for (int i = 0; i < M; i++) {
            int size = baseSize + (i < remainder ? 1 : 0);
            sizes[i] = new int[]{size, offset};
            offset += size;
        }
        return sizes;
    }

    VectorFloat<?> reusablePartialSums() {
        return partialSums.get();
    }

    AtomicReference<VectorFloat<?>> partialSquaredMagnitudes() {
        return partialSquaredMagnitudes;
    }

    AtomicReference<VectorFloat<?>> fusedPartialSquaredMagnitudes() {
        return fusedPartialSquaredMagnitudes;
    }

    public void write(IndexWriter out, int version) throws IOException
    {
        if (version > OnDiskGraphIndex.CURRENT_VERSION) {
            throw new IllegalArgumentException("Unsupported serialization version " + version);
        }
        if (version < 3 && anisotropicThreshold != UNWEIGHTED) {
            throw new IllegalArgumentException("Anisotropic threshold is only supported in serialization version 3 and above");
        }

        if (version >= 3) {
            out.writeInt(MAGIC);
            out.writeInt(version);
        }

        if (globalCentroid == null) {
            out.writeInt(0);
        } else {
            out.writeInt(globalCentroid.length());
            vectorTypeSupport.writeFloatVector(out, globalCentroid);
        }

        out.writeInt(M);
        assert Arrays.stream(subvectorSizesAndOffsets).mapToInt(m -> m[0]).sum() == originalDimension;
        assert M == subvectorSizesAndOffsets.length;
        for (var a : subvectorSizesAndOffsets) {
            out.writeInt(a[0]);
        }

        if (version >= 3) {
            out.writeFloat(anisotropicThreshold);
        }

        out.writeInt(clusterCount);
        // Keep the existing centroid-major binary format, regardless of the
        // in-memory representation. No second complete collection is built.
        for (int m = 0; m < M; m++) {
            if (!isTransposed()) {
                vectorTypeSupport.writeFloatVector(out, codebooks[m]);
                continue;
            }
            for (int c = 0; c < clusterCount; c++) {
                for (int d = 0; d < subvectorWidths[m]; d++) {
                    out.writeFloat(coordinate(m, c, d));
                }
            }
        }
    }

    /**
     * Creates a vector to hold partial sums for a single codebook.
     * The partial sums are the dot products of each subvector centroid in the codebook with the other subvector centroids.
     * Since the dot product is commutative, we only need to store the upper triangle of the matrix.
     * There are M codebooks, and each codebook has k centroids, so the total number of partial sums is M * k * (k+1) / 2.
     *
     * @return a vector to hold partial sums for a single codebook
     */
    public VectorFloat<?> createCodebookPartialSums(VectorSimilarityFunction vectorSimilarityFunction) {
        VectorFloat<?> partialSums = vectorTypeSupport.createFloatVector(getSubspaceCount() * getClusterCount() * (getClusterCount() + 1) / 2);
        int index = 0;
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < clusterCount; i++) {
                for (int j = i; j < clusterCount; j++) {

                    float sum = vectorSimilarityFunction == VectorSimilarityFunction.EUCLIDEAN ?
                            centroidDistance(m, i, j) :
                            centroidDot(m, i, j);

                    partialSums.set(index++, sum);
                }
            }
        }

        return partialSums;
    }

    @Override
    public int compressorSize() {
        int size = 0;
        size += Integer.BYTES; // MAGIC
        size += Integer.BYTES; // STORAGE_VERSION
        size += Integer.BYTES; // globalCentroidLength
        if (globalCentroid != null) {
            size += Float.BYTES * globalCentroid.length();
        }
        size += Integer.BYTES; // M
        size += Integer.BYTES * M; // subvectorSizesAndOffsets (only the sizes are written)
        size += Float.BYTES; // anisotropicThreshold
        size += Integer.BYTES; // clusterCount
        size += Float.BYTES * originalDimension * clusterCount;
        return size;
    }

    public static ProductQuantization load(RandomAccessReader in) throws IOException {
        return load(in, false);
    }

    /** Prepare an immutable in-memory layout for cosine or dot-product search. */
    public static ProductQuantization loadForSearch(RandomAccessReader in, VectorSimilarityFunction similarity) throws IOException {
        return load(in, similarity == VectorSimilarityFunction.COSINE || similarity == VectorSimilarityFunction.DOT_PRODUCT);
    }

    /**
     * Read the unchanged binary format into the provider's preferred search
     * layout. Unsupported shapes retain the original bulk path. This does not
     * mutate a compressor shared by another reader or retain a second copy.
     */
    static ProductQuantization load(RandomAccessReader in, boolean prepareForSearch) throws IOException {
        int maybeMagic = in.readInt();
        int version;
        int globalCentroidLength;
        if (maybeMagic != MAGIC) {
            // JVector 1+2 format, no magic or version, starts straight off with the centroid length
            version = 0;
            globalCentroidLength = maybeMagic;
        } else {
            version = in.readInt();
            globalCentroidLength = in.readInt();
        }

        VectorFloat<?> globalCentroid = null;
        if (globalCentroidLength > 0) {
            globalCentroid = vectorTypeSupport.readFloatVector(in, globalCentroidLength);
        }

        int M = in.readInt();
        int[][] subvectorSizes = new int[M][];
        int offset = 0;
        for (int i = 0; i < M; i++) {
            subvectorSizes[i] = new int[2];
            int size = in.readInt();
            subvectorSizes[i][0] = size;
            subvectorSizes[i][1] = offset;
            offset += size;
        }

        float anisotropicThreshold;
        if (version < 3) {
            anisotropicThreshold = UNWEIGHTED;
        } else {
            anisotropicThreshold = in.readFloat();
        }

        int clusters = in.readInt();
        if (prepareForSearch && VectorUtil.supportsTransposedPQ(subvectorSizes, clusters)) {
            var coordinates = vectorTypeSupport.createFloatVector(Math.multiplyExact(offset, clusters));
            for (int m = 0; m < M; m++) {
                int width = subvectorSizes[m][0];
                int start = subvectorSizes[m][1] * clusters;
                for (int c = 0; c < clusters; c++) {
                    for (int d = 0; d < width; d++) {
                        coordinates.set(start + d * clusters + c, in.readFloat());
                    }
                }
            }
            return new ProductQuantization(null, coordinates, clusters, subvectorSizes, globalCentroid, anisotropicThreshold);
        }
        VectorFloat<?>[] codebooks = new VectorFloat<?>[M];
        for (int m = 0; m < M; m++) {
            codebooks[m] = vectorTypeSupport.readFloatVector(in, clusters * subvectorSizes[m][0]);
        }
        return new ProductQuantization(codebooks, clusters, subvectorSizes, globalCentroid, anisotropicThreshold);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ProductQuantization that = (ProductQuantization) o;
        return M == that.M
                && originalDimension == that.originalDimension
                && Objects.equals(globalCentroid, that.globalCentroid)
                && Arrays.deepEquals(subvectorSizesAndOffsets, that.subvectorSizesAndOffsets)
                && codebooksEqual(that)
                && anisotropicThreshold == that.anisotropicThreshold;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(M, originalDimension);
        result = 31 * result + codebookHash();
        result = 31 * result + Objects.hashCode(globalCentroid);
        result = 31 * result + Arrays.deepHashCode(subvectorSizesAndOffsets);
        return result;
    }

    /**
     * @return the centroid of the codebooks
     */
    public VectorFloat<?> getOrComputeCentroid() {
        if (globalCentroid != null) {
            return globalCentroid;
        }

        // typically we only precompute the centroid for Euclidean similarity
        var centroid = vectorTypeSupport.createFloatVector(originalDimension);
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < clusterCount; i++) {
                var subspaceSize = subvectorSizesAndOffsets[m][0];
                var subCentroid = vectorTypeSupport.createFloatVector(subspaceSize);
                copyCentroid(m, i, subCentroid, 0);
                // we don't have addInPlace for vectors of different length, so do it by hand
                for (int j = 0; j < subspaceSize; j++) {
                    var k = subvectorSizesAndOffsets[m][1] + j;
                    centroid.set(k, centroid.get(k) + subCentroid.get(j));
                }
            }
        }
        VectorUtil.scale(centroid, 1.0f / M);
        return centroid;
    }

    @Override
    public int compressedVectorSize() {
        return M;
    }

    @Override
    public long ramBytesUsed() {
        long size = 0;
        if (transposedCodebooks != null) {
            size += transposedCodebooks.ramBytesUsed();
        } else {
            for (VectorFloat<?> codebook : codebooks) size += codebook.ramBytesUsed();
        }
        size += 32L + 8L * M; // immutable width and offset arrays
        size += 16L + 8L * M + M * (16L + 4L * clusterCount); // centroid norms

        return size;
    }


    /** Whether this search compressor retains its coordinates in transposed order. */
    public boolean isTransposed() {
        return transposedCodebooks != null;
    }

    /**
     * Return a centroid-major compressor for sustained encoding and graph building.
     * Search coordinates are immutable: conversion neither changes this compressor
     * nor caches a second representation in it. The caller owns the returned copy.
     * A compressor already in encoding order is returned unchanged.
     */
    public ProductQuantization forEncoding() {
        if (!isTransposed()) return this;
        VectorFloat<?>[] original = new VectorFloat<?>[M];
        for (int m = 0; m < M; m++) original[m] = centroidMajorCodebook(m);
        return new ProductQuantization(original, clusterCount, subvectorSizesAndOffsets,
                                       globalCentroid, anisotropicThreshold);
    }

    private float coordinate(int m, int c, int d) {
        return isTransposed() ? transposedCodebooks.get(subvectorOffsets[m] * clusterCount + d * clusterCount + c)
                              : codebooks[m].get(c * subvectorWidths[m] + d);
    }

    void copyCentroid(int m, int c, VectorFloat<?> target, int offset) {
        if (!isTransposed()) {
            target.copyFrom(codebooks[m], c * subvectorWidths[m], offset, subvectorWidths[m]);
        } else {
            for (int d = 0; d < subvectorWidths[m]; d++) target.set(offset + d, coordinate(m, c, d));
        }
    }

    private VectorFloat<?> centroidMajorCodebook(int m) {
        if (!isTransposed()) return codebooks[m];
        var result = vectorTypeSupport.createFloatVector(subvectorWidths[m] * clusterCount);
        for (int c = 0; c < clusterCount; c++) copyCentroid(m, c, result, c * subvectorWidths[m]);
        return result;
    }

    float centroidDot(int m, int c, VectorFloat<?> query, int offset) {
        if (!isTransposed()) return VectorUtil.dotProduct(codebooks[m], c * subvectorWidths[m], query, offset, subvectorWidths[m]);
        var scratch = centroidScratch.get()[0];
        copyCentroid(m, c, scratch, 0);
        return VectorUtil.dotProduct(scratch, 0, query, offset, subvectorWidths[m]);
    }

    float centroidDot(int m, int a, int b) {
        if (!isTransposed()) return VectorUtil.dotProduct(codebooks[m], a * subvectorWidths[m], codebooks[m], b * subvectorWidths[m], subvectorWidths[m]);
        var scratch = centroidScratch.get();
        copyCentroid(m, a, scratch[0], 0);
        copyCentroid(m, b, scratch[1], 0);
        return VectorUtil.dotProduct(scratch[0], 0, scratch[1], 0, subvectorWidths[m]);
    }

    float centroidDistance(int m, int c, VectorFloat<?> query, int offset) {
        if (!isTransposed()) return VectorUtil.squareL2Distance(codebooks[m], c * subvectorWidths[m], query, offset, subvectorWidths[m]);
        var scratch = centroidScratch.get()[0];
        copyCentroid(m, c, scratch, 0);
        return VectorUtil.squareL2Distance(scratch, 0, query, offset, subvectorWidths[m]);
    }

    float centroidDistance(int m, int a, int b) {
        if (!isTransposed()) return VectorUtil.squareL2Distance(codebooks[m], a * subvectorWidths[m], codebooks[m], b * subvectorWidths[m], subvectorWidths[m]);
        var scratch = centroidScratch.get();
        copyCentroid(m, a, scratch[0], 0);
        copyCentroid(m, b, scratch[1], 0);
        return VectorUtil.squareL2Distance(scratch[0], 0, scratch[1], 0, subvectorWidths[m]);
    }

    float centroidNorm(int m, int c) {
        return centroidNormsSquared[m][c];
    }

    void calculatePartialSums(VectorFloat<?> centeredQuery, VectorSimilarityFunction similarity, VectorFloat<?> output) {
        if (isTransposed() && similarity == VectorSimilarityFunction.DOT_PRODUCT) {
            VectorUtil.calculateTransposedPQDotProducts(transposedCodebooks, subvectorWidths, subvectorOffsets,
                                                       clusterCount, centeredQuery, output);
            return;
        }
        for (int m = 0; m < M; m++) {
            if (!isTransposed()) {
                VectorUtil.calculatePartialSums(codebooks[m], m, subvectorWidths[m], clusterCount,
                                                centeredQuery, subvectorOffsets[m], similarity, output);
            } else {
                // Euclidean search keeps the original distance calculation
                // rather than introducing cancellation through norm identities.
                for (int c = 0; c < clusterCount; c++) {
                    output.set(m * clusterCount + c, centroidDistance(m, c, centeredQuery, subvectorOffsets[m]));
                }
            }
        }
    }

    VectorFloat<?> createPartialSquaredMagnitudes() {
        var result = vectorTypeSupport.createFloatVector(M * clusterCount);
        for (int m = 0; m < M; m++) {
            // Preserve the original provider's cosine magnitude arithmetic. A
            // search layout needs only one temporary subspace, not a full copy.
            VectorUtil.calculatePartialSelfMagnitudes(centroidMajorCodebook(m), m,
                    subvectorWidths[m], clusterCount, result);
        }
        return result;
    }

    VectorFloat<?> createFusedPartialSquaredMagnitudes() {
        // Fused cosine historically used dotProduct per centroid, while the
        // standalone decoder used calculatePartialSelfMagnitudes. These can
        // differ by an ulp under a native provider, so keep separate caches.
        var result = vectorTypeSupport.createFloatVector(M * clusterCount);
        for (int m = 0; m < M; m++) {
            for (int c = 0; c < clusterCount; c++) {
                result.set(m * clusterCount + c, centroidNormsSquared[m][c]);
            }
        }
        return result;
    }

    private boolean codebooksEqual(ProductQuantization that) {
        for (int m = 0; m < M; m++) {
            if (!centroidMajorCodebook(m).equals(that.centroidMajorCodebook(m))) return false;
        }
        return true;
    }

    private int codebookHash() {
        int hash = 1;
        for (int m = 0; m < M; m++) hash = 31 * hash + centroidMajorCodebook(m).hashCode();
        return hash;
    }

    @Override
    public String toString() {
        if (anisotropicThreshold == UNWEIGHTED) {
            return String.format("ProductQuantization(M=%d, clusters=%d, centered=%s)", M, clusterCount, globalCentroid != null);
        }
        return String.format("ProductQuantization(M=%d, clusters=%d, centered=%s, anisotropicT=%.3f, eta=%.1f)",
                             M,
                             clusterCount,
                             globalCentroid != null,
                             anisotropicThreshold,
                             KMeansPlusPlusClusterer.computeParallelCostMultiplier(anisotropicThreshold, originalDimension));
    }

    private static void checkClusterCount(int clusterCount) {
        if (clusterCount > 256) {
            throw new IllegalArgumentException("Too many PQ clusters: " + clusterCount + " > 256");
        }
        if (clusterCount < 256) {
            LOG.warning("Using less than 256 PQ clusters will not reduce the memory footprint.");
        }
    }

    public int getOriginalDimension() {
        return originalDimension;
    }

    @Override
    public double reconstructionError(VectorFloat<?> vector) {
        var code = vectorTypeSupport.createByteSequence(M);

        if (globalCentroid != null) {
            vector = sub(vector, globalCentroid);
        }

        if (anisotropicThreshold > UNWEIGHTED)
            encodeAnisotropic(vector, code);
        else
            encodeUnweighted(vector, code);

        float sum = 0;
        for (int m = 0; m < M; m++) {
            int centroidIndex = Byte.toUnsignedInt(code.get(m));
            int centroidOffset = subvectorSizesAndOffsets[m][1];
            sum += centroidDistance(m, centroidIndex, vector, centroidOffset);
        }

        return sum / vector.length();
    }
}
