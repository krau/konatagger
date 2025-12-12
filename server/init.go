package server

import (
	"fmt"
	"path/filepath"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

func Init() error {
	onnxPath := filepath.Join(config.C().ModelDir, config.C().ModelFileName)
	tags, err := ReadTags(filepath.Join(config.C().ModelDir, config.C().ModelTagsName))
	if err != nil {
		return fmt.Errorf("failed to read tags: %w", err)
	}

	inputs, outputs, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		return fmt.Errorf("failed to get model input/output info: %w", err)
	}

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

	model = &Model{
		session:    session,
		input:      inputTensor,
		output:     outputTensor,
		inputName:  inputs[0].Name,
		outputName: outputs[0].Name,
		topTags:    tags,
	}
	return nil
}
