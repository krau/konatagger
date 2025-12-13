package service

import (
	"fmt"
	"image"
	"sort"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

func ModelPredict(img image.Image) (*PredictionResult, error) {
	inputData, err := Preprocess(img)
	if err != nil {
		return nil, err
	}
	if modelPool == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	m := <-modelPool
	defer func() { modelPool <- m }()

	copy(m.input.(*ort.Tensor[float32]).GetData(), inputData)
	if err := m.session.Run(); err != nil {
		return nil, err
	}

	logitsTensor := m.output.(*ort.Tensor[float32]).GetData()
	logits := make([]float32, len(logitsTensor))
	copy(logits, logitsTensor)

	var items []TagScore
	for i, v := range logits {
		p := Sigmoid(v)
		if p > config.C().Threshold {
			items = append(items, TagScore{
				Tag:   m.topTags[i],
				Score: p,
			})
		}
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].Score > items[j].Score
	})

	predicted := make([]string, 0, len(items))
	scores := make(map[string]float32, len(items))
	for _, it := range items {
		predicted = append(predicted, it.Tag)
		scores[it.Tag] = it.Score
	}

	result := &PredictionResult{
		PredictedTags: predicted,
		Scores:        scores,
	}

	return result, nil
}
