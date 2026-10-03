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

package io.github.jbellis.jvector.vector.cnative;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

public class LibraryLoaderTest {
    @Test
    public void selectsHostPlatformResources() {
        String oldOS = System.getProperty("os.name");
        String oldArch = System.getProperty("os.arch");
        try {
            for (String os : new String[] {"Linux", "Mac OS X", "Darwin"}) {
                System.setProperty("os.name", os);
                String ext = os.equals("Linux") ? ".so" : ".dylib";
                for (String arch : new String[] {"aarch64", "arm64"}) {
                    System.setProperty("os.arch", arch);
                    assertEquals("/libjvector-aarch64" + ext, LibraryLoader.resourceNameForArch());
                }
                for (String arch : new String[] {"amd64", "x86_64"}) {
                    System.setProperty("os.arch", arch);
                    assertEquals("/libjvector-x86_64" + ext, LibraryLoader.resourceNameForArch());
                }
            }
            System.setProperty("os.name", "Windows 11");
            assertNull(LibraryLoader.resourceNameForArch());
            System.setProperty("os.name", "Linux");
            System.setProperty("os.arch", "riscv64");
            assertNull(LibraryLoader.resourceNameForArch());
        } finally {
            if (oldOS == null) System.clearProperty("os.name"); else System.setProperty("os.name", oldOS);
            if (oldArch == null) System.clearProperty("os.arch"); else System.setProperty("os.arch", oldArch);
        }
    }
}
