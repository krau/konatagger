package config

import (
	"flag"
	"os"
	"runtime"
	"sync"

	"github.com/pelletier/go-toml/v2"
)

type Config struct {
	Token        string  `toml:"token" mapstructure:"token"`
	Host         string  `toml:"host" mapstructure:"host"`
	Port         string  `toml:"port" mapstructure:"port"`
	Threshold    float32 `toml:"threshold" mapstructure:"threshold"`
	Workers      int     `toml:"workers" mapstructure:"workers"`
	MaxBatchSize int     `toml:"max_batch_size" mapstructure:"max_batch_size"`
	Libonnx      string  `toml:"libonnx" mapstructure:"libonnx"`
	CUDA         bool    `toml:"cuda" mapstructure:"cuda"`
	CUDADevice   int     `toml:"cuda_device" mapstructure:"cuda_device"`

	ModelUrl      string `toml:"model_url" mapstructure:"model_url"`
	ModelDir      string `toml:"model_dir" mapstructure:"model_dir"`
	ModelTagsUrl  string `toml:"model_tags_url" mapstructure:"model_tags_url"`
	ModelTagsName string `toml:"model_tags_name" mapstructure:"model_tags_name"`
	ModelFileName string `toml:"model_file_name" mapstructure:"model_file_name"`
}

var (
	cfg = Config{
		Token:         "",
		Host:          "0.0.0.0",
		Port:          "8000",
		Threshold:     0.4,
		Workers:       defaultWorkers(),
		MaxBatchSize:  16,
		ModelUrl:      "https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx?download=true",
		ModelTagsUrl:  "https://huggingface.co/fancyfeast/joytag/resolve/main/top_tags.txt?download=true",
		ModelDir:      "models",
		ModelTagsName: "top_tags.txt",
		ModelFileName: "model.onnx",
	}
	loadOnce sync.Once
)

func defaultWorkers() int {
	// Concurrency only raises throughput once there are enough hardware
	// threads; one slot per four threads, capped at four, is where the
	// measured throughput plateaus. GOMAXPROCS also respects CPU quotas.
	return min(max(runtime.GOMAXPROCS(0)/4, 1), 4)
}

func C() Config {
	loadOnce.Do(func() {
		if _, err := os.Stat("config.toml"); err == nil {
			data, err := os.ReadFile("config.toml")
			if err != nil {
				panic(err)
			}
			if err := toml.Unmarshal(data, &cfg); err != nil {
				panic(err)
			}
		}

		fs := flag.NewFlagSet("konatagger", flag.ContinueOnError)
		fs.SetOutput(os.Stderr)

		token := fs.String("token", cfg.Token, "auth token for requests")
		host := fs.String("host", cfg.Host, "server host")
		port := fs.String("port", cfg.Port, "server port")
		threshold := fs.Float64("threshold", float64(cfg.Threshold), "prediction threshold")
		libonnx := fs.String("libonnx", cfg.Libonnx, "onnx runtime shared library path")
		cuda := fs.Bool("cuda", cfg.CUDA, "use the CUDA execution provider")
		cudaDevice := fs.Int("cuda_device", cfg.CUDADevice, "CUDA device id")
		modelURL := fs.String("model_url", cfg.ModelUrl, "model download url")
		modelDir := fs.String("model_dir", cfg.ModelDir, "model directory")
		modelTagsURL := fs.String("model_tags_url", cfg.ModelTagsUrl, "model tags download url")
		modelTagsName := fs.String("model_tags_name", cfg.ModelTagsName, "model tags filename")
		modelFileName := fs.String("model_file_name", cfg.ModelFileName, "model file name")
		workers := fs.Int("workers", cfg.Workers, "number of model workers")
		maxBatchSize := fs.Int("max_batch_size", cfg.MaxBatchSize, "maximum images per batch request")

		_ = fs.Parse(os.Args[1:])

		fs.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "token":
				cfg.Token = *token
			case "host":
				cfg.Host = *host
			case "port":
				cfg.Port = *port
			case "workers":
				cfg.Workers = *workers
			case "max_batch_size":
				cfg.MaxBatchSize = *maxBatchSize
			case "threshold":
				cfg.Threshold = float32(*threshold)
			case "libonnx":
				cfg.Libonnx = *libonnx
			case "cuda":
				cfg.CUDA = *cuda
			case "cuda_device":
				cfg.CUDADevice = *cudaDevice
			case "model_url":
				cfg.ModelUrl = *modelURL
			case "model_dir":
				cfg.ModelDir = *modelDir
			case "model_tags_url":
				cfg.ModelTagsUrl = *modelTagsURL
			case "model_tags_name":
				cfg.ModelTagsName = *modelTagsName
			case "model_file_name":
				cfg.ModelFileName = *modelFileName
			}
		})
		if cfg.Workers < 1 {
			cfg.Workers = defaultWorkers()
		}
		if cfg.MaxBatchSize < 1 {
			cfg.MaxBatchSize = 16
		}
	})
	return cfg
}
