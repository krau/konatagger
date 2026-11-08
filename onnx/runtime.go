package onnx

import (
	"embed"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/krau/konatagger/config"
)

//go:embed libs/*
var onnxLibs embed.FS

var pathOnce sync.Once
var libPath string

func LibPath() string {
	pathOnce.Do(func() {
		libPath = loadLibPath()
		if libPath == "" {
			slog.Error("ONNX Runtime library path could not be determined for this OS")
		} else {
			slog.Info("Using ONNX Runtime library", slog.String("path", libPath))
		}
	})
	return libPath
}

func loadLibPath() string {
	cfg := config.C()
	if cfg.CUDA {
		preloadCUDNN()
	}
	if cfg.Libonnx != "" {
		return cfg.Libonnx
	}
	if cfg.CUDA {
		if path := systemLibPath(); path != "" {
			slog.Info("Using system ONNX Runtime library for CUDA", slog.String("path", path))
			return path
		}
		slog.Warn("CUDA is enabled but no system ONNX Runtime library was found; falling back to the bundled CPU library")
	}
	switch runtime.GOOS {
	case "linux":
		path := filepath.Join("onnxlibs", "libonnxruntime-linux-x64.so.1.31.1")
		if _, err := os.Stat(path); err == nil {
			return path
		}
		if err := os.MkdirAll("onnxlibs", 0755); err != nil {
			slog.Error("Failed to create onnxlibs directory", slog.String("error", err.Error()))
			return ""
		}
		data, err := onnxLibs.ReadFile("libs/libonnxruntime-linux-x64.so.1.31.1")
		if err != nil {
			slog.Error("Failed to read embedded ONNX Runtime library", slog.String("error", err.Error()))
			return ""
		}
		if err := os.WriteFile(path, data, 0755); err != nil {
			slog.Error("Failed to write ONNX Runtime library to file", slog.String("error", err.Error()))
			return ""
		}
		return path
	case "darwin":
		return "/usr/local/lib/libonnxruntime.dylib"
	default:
		return ""
	}
}

func systemLibPath() string {
	candidates := []string{
		"/usr/local/lib/libonnxruntime.so",
		"/usr/lib/libonnxruntime.so",
	}
	switch runtime.GOARCH {
	case "amd64":
		candidates = append(candidates, "/usr/lib/x86_64-linux-gnu/libonnxruntime.so")
	case "arm64":
		candidates = append(candidates, "/usr/lib/aarch64-linux-gnu/libonnxruntime.so")
	}
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}
