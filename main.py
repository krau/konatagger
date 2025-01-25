from fastapi import FastAPI, File, Header, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from typing import List, Dict
import config
from predict import predict

app = FastAPI(title="KonaTagger", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def authenticate(auth_header: str = Header(...)):
    if auth_header != "Bearer " + config.config.get("token"):
        raise HTTPException(
            status_code=401, detail="认证失败", headers={"WWW-Authenticate": "Basic"}
        )


@app.post(
    "/predict",
    response_model=dict[str, List[str] | Dict[str, float]],
    summary="图像标注接口",
    description="接收图像并返回预测的标签和置信度分数",
)
async def predict_image(file: UploadFile = File(...), Authorization: str = Header(...)):
    authenticate(Authorization)
    if not file.content_type:
        raise HTTPException(status_code=400, detail="文件类型未知")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="仅支持图片文件上传")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predicted_tags, scores = await predict(image)

        scores = {k: float(v) for k, v in scores.items() if k in predicted_tags}
        sorted_scores = dict(
            sorted(scores.items(), key=lambda item: item[1], reverse=True)
        )

        return {"predicted_tags": predicted_tags, "scores": sorted_scores}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理过程中发生错误: {str(e)}")
    finally:
        await file.close()


@app.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.config.get("host", "0.0.0.0"),
        port=config.config.get("port", 8000),
    )
