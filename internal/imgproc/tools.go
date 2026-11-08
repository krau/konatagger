// Ported from github.com/disintegration/imaging v1.6.2.
// Copyright (c) 2012 Grigory Dryapak. MIT licensed; see LICENSE.

package imgproc

import (
	"bytes"
	"image"
	"image/color"
)

// New creates a new image with the specified width and height, and fills it
// with the specified color.
func New(width, height int, fillColor color.Color) *image.NRGBA {
	if width <= 0 || height <= 0 {
		return &image.NRGBA{}
	}

	c := color.NRGBAModel.Convert(fillColor).(color.NRGBA)
	if c == (color.NRGBA{0, 0, 0, 0}) {
		return image.NewNRGBA(image.Rect(0, 0, width, height))
	}

	return &image.NRGBA{
		Pix:    bytes.Repeat([]byte{c.R, c.G, c.B, c.A}, width*height),
		Stride: 4 * width,
		Rect:   image.Rect(0, 0, width, height),
	}
}

// Paste pastes img onto background at pos and returns the combined image.
func Paste(background, img image.Image, pos image.Point) *image.NRGBA {
	dst := Clone(background)
	pos = pos.Sub(background.Bounds().Min)
	pasteRect := image.Rectangle{Min: pos, Max: pos.Add(img.Bounds().Size())}
	interRect := pasteRect.Intersect(dst.Bounds())
	if interRect.Empty() {
		return dst
	}
	src := newScanner(img)
	parallel(interRect.Min.Y, interRect.Max.Y, func(ys <-chan int) {
		for y := range ys {
			x1 := interRect.Min.X - pasteRect.Min.X
			x2 := interRect.Max.X - pasteRect.Min.X
			y1 := y - pasteRect.Min.Y
			y2 := y1 + 1
			i1 := y*dst.Stride + interRect.Min.X*4
			i2 := i1 + interRect.Dx()*4
			src.scan(x1, y1, x2, y2, dst.Pix[i1:i2])
		}
	})
	return dst
}
