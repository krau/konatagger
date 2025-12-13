package service

import (
	"crypto/subtle"
	"errors"
	"image"
	"log/slog"

	"github.com/gin-gonic/gin"
	"github.com/krau/konatagger/config"
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

func HealthHandler(c *gin.Context) {
	c.JSON(200, gin.H{"status": "healthy"})
}
