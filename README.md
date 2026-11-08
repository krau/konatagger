# KonaTagger

Image tagging API.

Reference:
- [JoyTag](https://github.com/fpgaminer/joytag)

## Requirements

Onnx Runtime library: v1.31.1 (bundled CPU) and v1.29.0 (system CUDA) were tested on Linux/amd64.

If you are on Linux/amd64, the library has been embedded in the binary and will be extracted at runtime. For other platforms, please install it via your package manager or download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime)

### CUDA

To use a CUDA-enabled ONNX Runtime (e.g. the `onnxruntime-cuda` package), enable the execution provider:

```toml
cuda = true
cuda_device = 0
```

When `libonnx` is empty and `cuda` is enabled, the system library is looked up at `/usr/local/lib/libonnxruntime.so`, `/usr/lib/libonnxruntime.so`, and the Linux multiarch paths. The binding targets the ONNX Runtime 1.29 C API, so the system library must be 1.29 or newer.

## Configuration

Configuration file is in TOML format. Sample config file `config.example.toml` is provided.

Copy it to `config.toml` and modify as needed.

`workers` sets the number of concurrent inference requests. It defaults to 1-4 based on the CPU count.

`max_batch_size` limits the number of images accepted by `POST /predict/batch`. It defaults to 16.

## Usage

Client needs to add Authorization header with Bearer token.

```json
{
    "Authorization":"Bearer token"
}
```

### Routes

- `POST /predict`: Predict tags from an image.

Upload an image file via multipart/form-data, with key `file`.

Response:

```json
{
    "predicted_tags": ["tag1", "tag2", "tag3"],
    "scores": {
        "tag1": 0.9,
        "tag2": 0.8,
        "tag3": 0.7
    }
}
```

- `POST /predict/batch`: Predict tags for multiple images.

Upload up to `max_batch_size` images (default 16) via multipart/form-data, repeating the `file` key. Results are returned in the same order as the uploaded files.

Response:

```json
{
    "results": [
        {
            "predicted_tags": ["tag1", "tag2"],
            "scores": {"tag1": 0.9, "tag2": 0.8}
        },
        {
            "predicted_tags": ["tag3"],
            "scores": {"tag3": 0.7}
        }
    ]
}
```

- `GET /health`: Health check.

Response:

```json
{
    "status": "healthy"
}
```