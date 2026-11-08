// Ported from github.com/disintegration/imaging v1.6.2.
// Copyright (c) 2012 Grigory Dryapak. MIT licensed; see LICENSE.

package imgproc

import (
	"image"
	"math"
)

type indexWeight struct {
	index  int
	weight float64
}

const catmullRomSupport = 2.0

func bcspline(x, b, c float64) float64 {
	var y float64
	x = math.Abs(x)
	if x < 1.0 {
		y = ((12-9*b-6*c)*x*x*x + (-18+12*b+6*c)*x*x + (6 - 2*b)) / 6
	} else if x < 2.0 {
		y = ((-b-6*c)*x*x*x + (6*b+30*c)*x*x + (-12*b-48*c)*x + (8*b + 24*c)) / 6
	}
	return y
}

func catmullRomKernel(x float64) float64 {
	x = math.Abs(x)
	if x < catmullRomSupport {
		return bcspline(x, 0.0, 0.5)
	}
	return 0
}

func precomputeWeights(dstSize, srcSize int) [][]indexWeight {
	du := float64(srcSize) / float64(dstSize)
	scale := du
	if scale < 1.0 {
		scale = 1.0
	}
	ru := math.Ceil(scale * catmullRomSupport)

	out := make([][]indexWeight, dstSize)
	tmp := make([]indexWeight, 0, dstSize*int(ru+2)*2)

	for v := range dstSize {
		fu := (float64(v)+0.5)*du - 0.5

		begin := max(int(math.Ceil(fu-ru)), 0)
		end := min(int(math.Floor(fu+ru)), srcSize-1)

		var sum float64
		for u := begin; u <= end; u++ {
			w := catmullRomKernel((float64(u) - fu) / scale)
			if w != 0 {
				sum += w
				tmp = append(tmp, indexWeight{index: u, weight: w})
			}
		}
		if sum != 0 {
			for i := range tmp {
				tmp[i].weight /= sum
			}
		}

		out[v] = tmp
		tmp = tmp[len(tmp):]
	}

	return out
}

// Resize resizes the image to the specified width and height using the
// Catmull-Rom filter. If one of width or height is 0, the aspect ratio is
// preserved.
func Resize(img image.Image, width, height int) *image.NRGBA {
	dstW, dstH := width, height
	if dstW < 0 || dstH < 0 {
		return &image.NRGBA{}
	}
	if dstW == 0 && dstH == 0 {
		return &image.NRGBA{}
	}

	srcW := img.Bounds().Dx()
	srcH := img.Bounds().Dy()
	if srcW <= 0 || srcH <= 0 {
		return &image.NRGBA{}
	}

	if dstW == 0 {
		tmpW := float64(dstH) * float64(srcW) / float64(srcH)
		dstW = int(math.Max(1.0, math.Floor(tmpW+0.5)))
	}
	if dstH == 0 {
		tmpH := float64(dstW) * float64(srcH) / float64(srcW)
		dstH = int(math.Max(1.0, math.Floor(tmpH+0.5)))
	}

	if srcW != dstW && srcH != dstH {
		return resizeVertical(resizeHorizontal(img, dstW), dstH)
	}
	if srcW != dstW {
		return resizeHorizontal(img, dstW)
	}
	if srcH != dstH {
		return resizeVertical(img, dstH)
	}
	return Clone(img)
}

func resizeHorizontal(img image.Image, width int) *image.NRGBA {
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, width, src.h))
	weights := precomputeWeights(width, src.w)
	parallel(0, src.h, func(ys <-chan int) {
		scanLine := make([]uint8, src.w*4)
		for y := range ys {
			src.scan(0, y, src.w, y+1, scanLine)
			j0 := y * dst.Stride
			for x := range weights {
				var r, g, b, a float64
				for _, w := range weights[x] {
					i := w.index * 4
					s := scanLine[i : i+4 : i+4]
					aw := float64(s[3]) * w.weight
					r += float64(s[0]) * aw
					g += float64(s[1]) * aw
					b += float64(s[2]) * aw
					a += aw
				}
				if a != 0 {
					aInv := 1 / a
					j := j0 + x*4
					d := dst.Pix[j : j+4 : j+4]
					d[0] = clamp(r * aInv)
					d[1] = clamp(g * aInv)
					d[2] = clamp(b * aInv)
					d[3] = clamp(a)
				}
			}
		}
	})
	return dst
}

func resizeVertical(img image.Image, height int) *image.NRGBA {
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, src.w, height))
	weights := precomputeWeights(height, src.h)
	parallel(0, src.w, func(xs <-chan int) {
		scanLine := make([]uint8, src.h*4)
		for x := range xs {
			src.scan(x, 0, x+1, src.h, scanLine)
			for y := range weights {
				var r, g, b, a float64
				for _, w := range weights[y] {
					i := w.index * 4
					s := scanLine[i : i+4 : i+4]
					aw := float64(s[3]) * w.weight
					r += float64(s[0]) * aw
					g += float64(s[1]) * aw
					b += float64(s[2]) * aw
					a += aw
				}
				if a != 0 {
					aInv := 1 / a
					j := y*dst.Stride + x*4
					d := dst.Pix[j : j+4 : j+4]
					d[0] = clamp(r * aInv)
					d[1] = clamp(g * aInv)
					d[2] = clamp(b * aInv)
					d[3] = clamp(a)
				}
			}
		}
	})
	return dst
}

// Clone returns a copy of the given image.
func Clone(img image.Image) *image.NRGBA {
	src := newScanner(img)
	dst := image.NewNRGBA(image.Rect(0, 0, src.w, src.h))
	size := src.w * 4
	parallel(0, src.h, func(ys <-chan int) {
		for y := range ys {
			i := y * dst.Stride
			src.scan(0, y, src.w, y+1, dst.Pix[i:i+size])
		}
	})
	return dst
}
