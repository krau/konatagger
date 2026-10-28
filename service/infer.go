package service

import (
	"fmt"
	"image"
	"sort"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

func ModelPredict(img image.Image) (*PredictionResult, error) {
	if workerPool == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	m := <-workerPool
	defer func() { workerPool <- m }()

	if err := preprocessInto(img, m.input.GetData()); err != nil {
		return nil, err
	}
	if err := m.session.Run([]ort.Value{m.input}, []ort.Value{m.output}); err != nil {
		return nil, err
	}

	logits := m.output.GetData()

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
