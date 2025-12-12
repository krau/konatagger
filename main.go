package main

import (
	"log/slog"

	"github.com/gin-gonic/gin"
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

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.POST("/predict", server.PredictHandler)
	r.GET("/health", server.HealthHandler)

	addr := config.C().Host + ":" + config.C().Port
	slog.Info("Listening on", slog.String("address", addr))
	if err := r.Run(addr); err != nil {
		slog.Error("Failed to start HTTP server", slog.String("error", err.Error()))
	}
}
