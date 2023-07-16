from PIL import Image
import gradio as gr
import requests
import io
import numpy as np
import cv2
api_url = "http://10.1.38.211:5202/upload"
from io import BytesIO
def predict(image):
    # Convert to bytesio
    img_receive = io.BytesIO()
    image.save(img_receive, format="PNG")
    # receive response
    img_receive.seek(0)
    response = requests.post(api_url, data = {"id":20},files={"image": img_receive})
    img_bytes_io = io.BytesIO(response.content)
    
    result = Image.open(img_bytes_io)
    return result

if __name__ == '__main__':
    # img = PIL.Image.open('test.jpg')
    # predict(img)
    input = gr.inputs.Image(type="pil")
    demo = gr.interface.Interface(fn=predict, inputs=input, outputs="image")
    demo.launch(share=True)