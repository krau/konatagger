package server

import (
	"crypto/subtle"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"sort"

	"github.com/gin-gonic/gin"
	"github.com/krau/konatagger/config"
	ort "github.com/yalue/onnxruntime_go"
)

var (
	errUnauthorized = errors.New("unauthorized")
)

func authenticate(c *gin.Context) error {
	auth := c.GetHeader("Authorization")

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

func PredictHandler(c *gin.Context) {
	if err := authenticate(c); err != nil {
		c.JSON(401, gin.H{"error": "认证失败"})
		return
	}

	fileHeader, err := c.FormFile("file")
	if err != nil {
		c.JSON(400, gin.H{"error": "未上传文件"})
		return
	}

	file, err := fileHeader.Open()
	if err != nil {
		c.JSON(400, gin.H{"error": "无法打开上传的文件"})
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		c.JSON(400, gin.H{"error": "无法解析图片"})
		return
	}

	resp, err := ModelPredict(img)
	if err != nil {
		slog.Error("Prediction failed", slog.String("error", err.Error()))
		c.JSON(500, gin.H{"error": "推理失败"})
		return
	}

	c.JSON(200, resp)
}

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

func HealthHandler(c *gin.Context) {
	c.JSON(200, gin.H{"status": "healthy"})
}
