package server

import (
	"crypto/subtle"
	"encoding/json"
	"errors"
	"image"
	"io"
	"net/http"
	"sort"

	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	errUnauthorized = errors.New("unauthorized")
)

func authenticate(r *http.Request) error {
	auth := r.Header.Get("Authorization")

	expectedToken := config.C().Token
	if expectedToken == "" {
		return nil
	}
	providedToken := ""
	if len(auth) > 7 && auth[:7] == "Bearer " {
		providedToken = auth[7:]
	}
	if subtle.ConstantTimeCompare([]byte(providedToken), []byte(expectedToken)) != 1 {
		return errUnauthorized
	}

	return nil
}

func PredictHandler(w http.ResponseWriter, r *http.Request) {
	if err := authenticate(r); err != nil {
		http.Error(w, "认证失败", http.StatusUnauthorized)
		return
	}

	file, _, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "未上传文件", http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "无法解析图片", http.StatusBadRequest)
		return
	}

	inputData, err := Preprocess(img)
	if err != nil {
		http.Error(w, "预处理失败", http.StatusInternalServerError)
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	copy(model.input.(*ort.Tensor[float32]).GetData(), inputData)

	if err := model.session.Run(); err != nil {
		http.Error(w, "推理失败", http.StatusInternalServerError)
		return
	}

	logits := model.output.(*ort.Tensor[float32]).GetData()
	type pair struct {
		Tag   string
		Score float32
	}
	var items []pair
	for i, v := range logits {
		p := Sigmoid(v)
		if p > config.C().Threshold {
			items = append(items, pair{
				Tag:   model.topTags[i],
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

	resp := map[string]any{
		"predicted_tags": predicted,
		"scores":         scores,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func ModelPredict(img image.Image) (map[string]float32, error) {
	inputData, err := Preprocess(img)
	if err != nil {
		return nil, err
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	copy(model.input.(*ort.Tensor[float32]).GetData(), inputData)

	if err := model.session.Run(); err != nil {
		return nil, err
	}

	logits := model.output.(*ort.Tensor[float32]).GetData()
	scores := make(map[string]float32)
	for i, v := range logits {
		p := Sigmoid(v)
		if p > config.C().Threshold {
			scores[model.topTags[i]] = p
		}
	}

	return scores, nil
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"status":"healthy"}`)
}
