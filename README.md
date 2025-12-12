# KonaTagger

Image tagging API.

Reference:
- [JoyTag](https://github.com/fpgaminer/joytag)

## Requirements

Onnx Runtime library: Only v1.23.2 was tested on Linux/amd64.

If you are on Linux/amd64, the library has been embedded in the binary and will be extracted at runtime. For other platforms, please install it via your package manager or download from [ONNX Runtime releases](https://github.com/microsoft/onnxruntime)

## Configuration

Configuration file is in TOML format. Sample config file `config.example.toml` is provided.

Copy it to `config.toml` and modify as needed.

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

- `GET /health`: Health check.

Response:

```json
{
    "status": "healthy"
}
```