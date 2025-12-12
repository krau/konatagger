package config

import (
	"flag"
	"os"
	"sync"

	"github.com/pelletier/go-toml/v2"
)

type Config struct {
	Token     string  `toml:"token" mapstructure:"token"`
	Host      string  `toml:"host" mapstructure:"host"`
	Port      string  `toml:"port" mapstructure:"port"`
	Threshold float32 `toml:"threshold" mapstructure:"threshold"`
	Libonnx   string  `toml:"libonnx" mapstructure:"libonnx"`

	ModelUrl      string `toml:"model_url" mapstructure:"model_url"`
	ModelDir      string `toml:"model_dir" mapstructure:"model_dir"`
	ModelTagsName string `toml:"model_tags_name" mapstructure:"model_tags_name"`
	ModelFileName string `toml:"model_file_name" mapstructure:"model_file_name"`
}

var (
	cfg = Config{
		Token:         "",
		Host:          "0.0.0.0",
		Port:          "8000",
		Threshold:     0.4,
		ModelUrl:      "https://huggingface.co/fancyfeast/joytag/resolve/main/model.onnx?download=true",
		ModelDir:      "models",
		ModelTagsName: "top_tags.txt",
		ModelFileName: "model.onnx",
	}
	loadOnce sync.Once
)

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
		modelURL := fs.String("model_url", cfg.ModelUrl, "model download url")
		modelDir := fs.String("model_dir", cfg.ModelDir, "model directory")
		modelTagsName := fs.String("model_tags_name", cfg.ModelTagsName, "model tags filename")
		modelFileName := fs.String("model_file_name", cfg.ModelFileName, "model file name")

		_ = fs.Parse(os.Args[1:])

		fs.Visit(func(f *flag.Flag) {
			switch f.Name {
			case "token":
				cfg.Token = *token
			case "host":
				cfg.Host = *host
			case "port":
				cfg.Port = *port
			case "threshold":
				cfg.Threshold = float32(*threshold)
			case "libonnx":
				cfg.Libonnx = *libonnx
			case "model_url":
				cfg.ModelUrl = *modelURL
			case "model_dir":
				cfg.ModelDir = *modelDir
			case "model_tags_name":
				cfg.ModelTagsName = *modelTagsName
			case "model_file_name":
				cfg.ModelFileName = *modelFileName
			}
		})
	})
	return cfg
}
