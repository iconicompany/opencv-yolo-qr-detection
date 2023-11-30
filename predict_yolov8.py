from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/detect/train3/weights/best.onnx')

res = model("<image>")

for r in res:
    # print("[RESULT]: \n", r.boxes.xywh)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])  
    im.save('results.jpg') # image with outline dtmx
res[0].save_crop(save_dir="crop", file_name="") # only dtmx
res[0].boxes.xywh.tolist()[0] # box with dtmx
