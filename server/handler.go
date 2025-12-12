package server

import (
	"crypto/subtle"
	"errors"
	"image"
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

	inputData, err := Preprocess(img)
	if err != nil {
		c.JSON(500, gin.H{"error": "预处理失败"})
		return
	}

	model.mu.Lock()
	defer model.mu.Unlock()

	copy(model.input.(*ort.Tensor[float32]).GetData(), inputData)

	if err := model.session.Run(); err != nil {
		c.JSON(500, gin.H{"error": "推理失败"})
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

	c.JSON(200, resp)
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

func HealthHandler(c *gin.Context) {
	c.JSON(200, gin.H{"status": "healthy"})
}
