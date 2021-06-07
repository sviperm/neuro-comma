from typing import List, Union

from fastapi import FastAPI
from predict import Predictor
from pydantic import BaseModel

app = FastAPI()

punct_restorer = Predictor('repunct-model^4', model_weights='weights_ep4_9910.pt')


class Data(BaseModel):
    data: Union[str, List[str]]


class InputData(Data):
    class Config:
        schema_extra = {
            "example": {
                "data": ("Сделать модель восстановления знаков "
                         "препинания у текста который будет получен "
                         "с помощью Speech-to-Text сервиса")
            }
        }


class OutputData(Data):
    class Config:
        schema_extra = {
            "example": {
                "data": ("Сделать модель восстановления знаков "
                         "препинания у текста, который будет получен "
                         "с помощью Speech-to-Text сервиса")
            }
        }


@app.post("/", response_model=OutputData)
async def punctuation_restoration(input: InputData):
    data = input.data
    if isinstance(data, str):
        output_data = punct_restorer(data)
    else:
        output_data = [punct_restorer(text) for text in data]
    return {"data": output_data}


@app.post("/clear_commas", response_model=OutputData)
async def remove_comas_punctuation_restoration(input: InputData):
    data = input.data
    if isinstance(data, str):
        output_data = punct_restorer(data.replace(',', ''))
    else:
        output_data = [punct_restorer(text.replace(',', '')) for text in data]
    return {"data": output_data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
