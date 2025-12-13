package service

import (
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"

	_ "github.com/gen2brain/avif"
	_ "golang.org/x/image/webp"

	"strings"

	"github.com/disintegration/imaging"
)

func Sigmoid(x float32) float32 {
	if x > 50 {
		x = 50
	} else if x < -50 {
		x = -50
	}
	return 1 / (1 + float32(math.Exp(float64(-x))))
}

func ReadLines(path string) ([]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	lines := strings.Split(string(b), "\n")
	var tags []string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if l != "" {
			tags = append(tags, l)
		}
	}
	return tags, nil
}

// prepare image for model input
func Preprocess(img image.Image) ([]float32, error) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	maxDim := max(h, w)

	// white padding
	canvas := imaging.New(maxDim, maxDim, color.White)
	img = imaging.Paste(canvas, img, image.Pt((maxDim-w)/2, (maxDim-h)/2))
	img = imaging.Resize(img, ImageSize, ImageSize, imaging.Lanczos)

	out := make([]float32, 3*ImageSize*ImageSize)
	rBase := 0
	gBase := ImageSize * ImageSize
	bBase := 2 * ImageSize * ImageSize

	for y := range ImageSize {
		for x := range ImageSize {
			r, g, b, _ := img.At(x, y).RGBA()
			fr := float32(r) / 65535.0
			fg := float32(g) / 65535.0
			fb := float32(b) / 65535.0

			out[rBase] = (fr - ClipMean[0]) / ClipStd[0]
			out[gBase] = (fg - ClipMean[1]) / ClipStd[1]
			out[bBase] = (fb - ClipMean[2]) / ClipStd[2]

			rBase++
			gBase++
			bBase++
		}
	}
	return out, nil
}
