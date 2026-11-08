// Ported from github.com/disintegration/imaging v1.6.2.
// Copyright (c) 2012 Grigory Dryapak. MIT licensed; see LICENSE.

package imgproc

import (
	"runtime"
	"sync"
)

func parallel(start, stop int, fn func(<-chan int)) {
	count := stop - start
	if count < 1 {
		return
	}

	procs := min(runtime.GOMAXPROCS(0), count)

	c := make(chan int, count)
	for i := start; i < stop; i++ {
		c <- i
	}
	close(c)

	var wg sync.WaitGroup
	for range procs {
		wg.Go(func() {
			fn(c)
		})
	}
	wg.Wait()
}

func clamp(x float64) uint8 {
	v := int64(x + 0.5)
	if v > 255 {
		return 255
	}
	if v > 0 {
		return uint8(v)
	}
	return 0
}
