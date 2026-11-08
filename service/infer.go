package service

import (
	"fmt"
	"image"
	"io"
	"sort"
	"sync"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

const batchPrepConcurrency = 2

type DecodeError struct {
	Index int
	Err   error
}

func (e *DecodeError) Error() string {
	return fmt.Sprintf("failed to decode image %d: %v", e.Index+1, e.Err)
}

func (e *DecodeError) Unwrap() error { return e.Err }

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

	return predictFromLogits(logits, topTags), nil
}

func ModelPredictBatch(readers []io.Reader) ([]*PredictionResult, error) {
	if workerPool == nil || preprocessPool == nil {
		return nil, fmt.Errorf("model not initialized")
	}
	n := len(readers)
	if n == 0 {
		return nil, fmt.Errorf("empty batch")
	}
	if maxBatch := config.C().MaxBatchSize; n > maxBatch {
		return nil, fmt.Errorf("batch size %d exceeds limit %d", n, maxBatch)
	}

	input := make([]float32, n*3*imagePixels)
	sem := make(chan struct{}, min(n, batchPrepConcurrency))
	var wg sync.WaitGroup
	var firstErr error
	var errOnce sync.Once
	for i, reader := range readers {
		wg.Add(1)
		go func(i int, reader io.Reader) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			img, _, err := image.Decode(reader)
			if err != nil {
				errOnce.Do(func() { firstErr = &DecodeError{Index: i, Err: err} })
				return
			}

			preprocessPool <- struct{}{}
			err = preprocessInto(img, input[i*3*imagePixels:(i+1)*3*imagePixels])
			<-preprocessPool
			if err != nil {
				errOnce.Do(func() { firstErr = fmt.Errorf("failed to preprocess image %d: %w", i+1, err) })
			}
		}(i, reader)
	}
	wg.Wait()
	if firstErr != nil {
		return nil, firstErr
	}

	m := <-workerPool
	in, err := ort.NewTensor(ort.NewShape(int64(n), 3, ImageSize, ImageSize), input)
	if err != nil {
		workerPool <- m
		return nil, fmt.Errorf("failed to create batch input tensor: %w", err)
	}
	defer in.Destroy()

	tagCount := len(m.topTags)
	out, err := ort.NewEmptyTensor[float32](ort.NewShape(int64(n), int64(tagCount)))
	if err != nil {
		workerPool <- m
		return nil, fmt.Errorf("failed to create batch output tensor: %w", err)
	}
	defer out.Destroy()

	if err := m.session.Run([]ort.Value{in}, []ort.Value{out}); err != nil {
		workerPool <- m
		return nil, fmt.Errorf("failed to run batch inference: %w", err)
	}
	logits := make([]float32, len(out.GetData()))
	copy(logits, out.GetData())
	topTags := m.topTags
	workerPool <- m

	results := make([]*PredictionResult, n)
	for i := range n {
		start := i * tagCount
		results[i] = predictFromLogits(logits[start:start+tagCount], topTags)
	}
	return results, nil
}

func predictFromLogits(logits []float32, topTags []string) *PredictionResult {
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

	return &PredictionResult{
		PredictedTags: predicted,
		Scores:        scores,
	}
}
