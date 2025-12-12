package config

import (
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
	})
	return cfg
}
