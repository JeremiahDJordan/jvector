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

package io.github.jbellis.jvector.vector;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TransposedPQProviderTest {
    @Test
    public void onlyNeonAndMeasuredShapesOptIn() {
        var provider = new NativeVectorizationProvider();
        var support = (NativeVectorUtilSupport) provider.getVectorUtilSupport();
        assertEquals("neon".equals(support.getActiveIsa()), support.supportsTransposedPQ(new int[][] {{1,0}, {8,1}},256));
        assertFalse(support.supportsTransposedPQ(new int[][] {{9,0}},256));
        assertFalse(support.supportsTransposedPQ(new int[][] {{0,0}},256));
        assertFalse(support.supportsTransposedPQ(new int[][] {{1,0}},17));
        assertFalse(support.supportsTransposedPQ(new int[0][],256));
        assertFalse(new DefaultVectorUtilSupport().supportsTransposedPQ(new int[][] {{1,0}},256));
        assertFalse(new PanamaVectorUtilSupport().supportsTransposedPQ(new int[][] {{1,0}},256));
    }
}
