package main

import (
	"context"
	"fmt"
	"image"
	"log/slog"
	"os"
	"os/signal"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/krau/konatagger/config"
	"github.com/krau/konatagger/onnx"
	"github.com/krau/konatagger/service"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	if len(os.Args) > 1 && os.Args[1] == "predict" {
		if len(os.Args) < 3 {
			slog.Error("missing image path argument for predict command")
			return
		}
		cmdPredictOnce(ctx, os.Args[2])
		return
	}

	slog.Info("Starting KonaTagger")
	ort.SetSharedLibraryPath(onnx.LibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		slog.Error("Failed to initialize ONNX Runtime environment", slog.String("error", err.Error()))
		return
	}
	defer ort.DestroyEnvironment()
	go func() {
		if err := service.Init(ctx); err != nil {
			slog.Error("Failed to initialize server", slog.String("error", err.Error()))
			return
		}

		gin.SetMode(gin.ReleaseMode)
		r := gin.Default()
		r.POST("/predict", service.PredictHandler)
		r.GET("/health", service.HealthHandler)

		addr := config.C().Host + ":" + config.C().Port
		slog.Info("Listening on", slog.String("address", addr))

		if err := r.Run(addr); err != nil {
			slog.Error("Server error", slog.String("error", err.Error()))
			cancel()
		}
	}()

	<-ctx.Done()
	slog.Info("shutting down")
}

func cmdPredictOnce(ctx context.Context, imgPath string) {
	ort.SetSharedLibraryPath(onnx.LibPath())
	if err := ort.InitializeEnvironment(); err != nil {
		slog.Error("Failed to initialize ONNX Runtime environment", slog.String("error", err.Error()))
		return
	}
	defer ort.DestroyEnvironment()
	if err := service.Init(ctx); err != nil {
		slog.Error("Failed to initialize", slog.String("error", err.Error()))
		return
	}
	file, err := os.Open(imgPath)
	if err != nil {
		slog.Error("Failed to open image", slog.String("error", err.Error()))
		return
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		slog.Error("Failed to decode image", slog.String("error", err.Error()))
		return
	}
	start := time.Now()
	resp, err := service.ModelPredict(img)
	if err != nil {
		slog.Error("Prediction failed", slog.String("error", err.Error()))
		return
	}
	fmt.Println("Predicted results (in", time.Since(start).String()+"):")
	for _, tag := range resp.PredictedTags {
		fmt.Printf("- %s: %.4f\n", tag, resp.Scores[tag])
	}
}
