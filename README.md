# KonaTagger

Image tagging API.

Reference:
- [JoyTag](https://github.com/fpgaminer/joytag)

## Installation

Requirements:

- Python 3.12+

Clone the repository.

Create a virtual environment and install the dependencies.

> If your machine does not have GPU, you can install the CPU version by option: --index-url https://download.pytorch.org/whl/cpu

Edit the configuration file `config.toml`:

```toml
token = "token" # Bearer token
port = 39917
host = "0.0.0.0"
threshold = 0.5
device = "cuda" # "cpu" or "cuda"
```

Run the server:

```bash
python main.py
```

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