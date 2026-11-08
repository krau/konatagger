//go:build linux && cgo

package onnx

/*
#cgo LDFLAGS: -ldl
#include <dlfcn.h>

static void preloadCUDNN(void) {
	const char *libs[] = {"libcudnn.so.9", "libcudnn.so", 0};
	for (int i = 0; libs[i]; i++) {
		if (dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL) != 0) {
			return;
		}
	}
}
*/
import "C"

// preloadCUDNN exposes cuDNN symbols globally before the CUDA execution
// provider is loaded. The provider shipped by some distributions references
// cuDNN without declaring it as a shared library dependency.
func preloadCUDNN() {
	C.preloadCUDNN()
}
