import sys

sys.path.append(".")
from server.logger_ser import logger
from PIL import Image
import onnxruntime
import numpy as np
import io


classes = {0: "bottom_left", 1: "bottom_right", 2: "top_left", 3 : 'top_right'}


class Classifier:
    def __init__(self):
        self.session = self.__load_onnx_session()

    def __load_onnx_session(
        self,
        onnx_path="/home/hai/workspace/api-server-template/onnx_checkpoints/last.onnx",
    ):
        provider = ["CPUExecutionProvider"]
        session = onnxruntime.InferenceSession(onnx_path, providers=provider)
        return session

    def onnx_classify(self, img_path, thresh=0.6):
        try:
            if isinstance(img_path, str):
                image_root = Image.open(img_path).convert("RGB")
            else:
                # img_bytes = img_path.read()
                # image_root = Image.open(io.BytesIO(img_path)).convert("RGB")
                image_root = Image.open(img_path).convert("RGB")

            image = image_root.copy().resize((512, 512), Image.BILINEAR)
            image = np.array(image)

            mean = np.array([0, 0, 0])
            std = np.array([255, 255, 255])
            image = (image - mean) / std

            image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
            image = np.expand_dims(image, axis=0)  # CHW -> BCHW
 
            image = np.float32(image)

            out = self.session.run(None, {self.session.get_inputs()[0].name: image})[0]
            out = self.__softmax(out)[0]

            idx = int(np.argmax(out))
            if out[idx] > thresh:
                return True, {
                    "type_code": idx,
                    "score": round(100 * out[idx].item(), 2),
                    "type_msg": classes[idx],
                }
            else:
                return True, {
                    "type_code": 2,
                    "score": round(100 * out[idx].item(), 2),
                    "type_msg": classes[2],
                }

        except Exception as e:
            logger.error(e)
            return False, None

    # for validating purpose, use this function in validation.py
    def onnx_validating(self, img_path, thresh=0.6):
        if type(img_path) == str:
            image_root = Image.open(img_path).convert("RGB")
        else:
            image_root = Image.open(io.BytesIO(img_path)).convert("RGB")

        image = image_root.copy().resize((512, 512), Image.BILINEAR)
        image = np.array(image)

        mean = np.array([0, 0, 0])
        std = np.array([255, 255, 255])
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = np.expand_dims(image, axis=0)  # CHW -> BCHW

        image = np.float32(image)

        out = self.session.run(None, {self.session.get_inputs()[0].name: image})[0]
        out = self.__softmax(out)[0]
        return out

    @staticmethod
    def __softmax(x):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()

