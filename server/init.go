package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

func ensureFile(ctx context.Context, path, url string, name string) error {
	if _, err := os.Stat(path); err == nil {
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to stat %s file: %w", name, err)
	}

	if url == "" {
		return fmt.Errorf("%s file does not exist and no download url is configured", name)
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create directory for %s file: %w", name, err)
	}
	slog.Info("Downloading model file", slog.String("name", name), slog.String("url", url))
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download %s file: %w", name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download %s file: unexpected status %s", name, resp.Status)
	}

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create %s file: %w", name, err)
	}
	defer f.Close()

	if _, err := io.Copy(f, resp.Body); err != nil {
		return fmt.Errorf("failed to write %s file: %w", name, err)
	}

	return nil
}

func Init(ctx context.Context) error {
	cfg := config.C()
	onnxPath := filepath.Join(cfg.ModelDir, cfg.ModelFileName)
	if err := ensureFile(ctx, onnxPath, cfg.ModelUrl, "model"); err != nil {
		return err
	}

	tagsPath := filepath.Join(cfg.ModelDir, cfg.ModelTagsName)
	if err := ensureFile(ctx, tagsPath, cfg.ModelTagsUrl, "tags"); err != nil {
		return err
	}

	tags, err := ReadTags(tagsPath)
	if err != nil {
		return fmt.Errorf("failed to read tags: %w", err)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		return fmt.Errorf("failed to get model input/output info: %w", err)
	}

	modelPool = make(chan *Model, cfg.Workers)
	for i := 0; i < cfg.Workers; i++ {
		opts, err := ort.NewSessionOptions()
		if err != nil {
			return fmt.Errorf("failed to create session options: %w", err)
		}
		inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, ImageSize, ImageSize), make([]float32, 3*ImageSize*ImageSize))
		if err != nil {
			return fmt.Errorf("failed to create input tensor: %w", err)
		}
		outputTensor, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(len(tags))))
		if err != nil {
			return fmt.Errorf("failed to create output tensor: %w", err)
		}

		session, err := ort.NewAdvancedSession(
			onnxPath,
			[]string{inputs[0].Name},
			[]string{outputs[0].Name},
			[]ort.Value{inputTensor},
			[]ort.Value{outputTensor},
			opts,
		)
		if err != nil {
			return fmt.Errorf("failed to create ONNX Runtime session: %w", err)
		}

		m := &Model{
			session:    session,
			input:      inputTensor,
			output:     outputTensor,
			inputName:  inputs[0].Name,
			outputName: outputs[0].Name,
			topTags:    tags,
		}
		modelPool <- m
	}
	return nil
}
