package service

import (
	"fmt"
	"image"
	"sort"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

func ModelPredict(img image.Image) (*PredictionResult, error) {
	if workerPool == nil || preprocessPool == nil {
		return nil, fmt.Errorf("model not initialized")
	}

	bufPtr := preprocessBuffers.Get().(*[]float32)
	defer preprocessBuffers.Put(bufPtr)
	buf := *bufPtr

	preprocessPool <- struct{}{}
	err := preprocessInto(img, buf)
	<-preprocessPool
	if err != nil {
		return nil, err
	}

	m := <-workerPool
	copy(m.input.GetData(), buf)
	if err := m.session.Run([]ort.Value{m.input}, []ort.Value{m.output}); err != nil {
		workerPool <- m
		return nil, err
	}
	logits := make([]float32, len(m.output.GetData()))
	copy(logits, m.output.GetData())
	topTags := m.topTags
	workerPool <- m

	var items []TagScore
	for i, v := range logits {
		p := Sigmoid(v)
		if p > config.C().Threshold {
			items = append(items, TagScore{
				Tag:   topTags[i],
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
