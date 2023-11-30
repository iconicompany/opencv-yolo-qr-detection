from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/detect/train2/weights/best.onnx')

# model.export(format="saved_model", keras=True)

res = model("<image>")

for r in res:
    # print("[RESULT]: \n", r.boxes.xywh)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])  
    im.save('results.jpg') # image with outline dtmx
res[0].save_crop(save_dir="crop", file_name="") # only dtmx
