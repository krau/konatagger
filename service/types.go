package service

import (
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

const ImageSize = 448

var (
	ClipMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	ClipStd  = [3]float32{0.26862954, 0.26130258, 0.27577711}
)

type TagScore struct {
	Tag   string  `json:"tag"`
	Score float32 `json:"score"`
}

type PredictionResult struct {
	PredictedTags []string           `json:"predicted_tags"`
	Scores        map[string]float32 `json:"scores"`
}

type Model struct {
	session    *ort.AdvancedSession
	input      ort.Value
	output     ort.Value
	inputName  string
	outputName string
	topTags    []string
	mu         sync.Mutex
}

var modelPool chan *Model
