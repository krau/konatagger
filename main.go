package main

import (
	"log/slog"
	"net/http"

	"github.com/krau/konatagger/config"
	"github.com/krau/konatagger/onnx"
	"github.com/krau/konatagger/server"

	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	slog.Info("Starting KonaTagger")

	ort.SetSharedLibraryPath(onnx.LibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		slog.Error("Failed to initialize ONNX Runtime environment", slog.String("error", err.Error()))
		return
	}
	defer ort.DestroyEnvironment()

	if err := server.Init(); err != nil {
		slog.Error("Failed to initialize server", slog.String("error", err.Error()))
		return
	}

	http.HandleFunc("/predict", server.PredictHandler)
	http.HandleFunc("/health", server.HealthHandler)

	addr := config.C().Host + ":" + config.C().Port
	slog.Info("Listening on", slog.String("address", addr))
	if err := http.ListenAndServe(addr, nil); err != nil {
		slog.Error("Failed to start HTTP server", slog.String("error", err.Error()))
	}
}
