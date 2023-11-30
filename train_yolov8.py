from ultralytics import YOLO

# model = YOLO('yolo/yolov8n.pt') # fast model
model = YOLO('yolo/yolov8x.pt') # accuracy model

results = model.train(data='data.yaml', epochs=1, imgsz=640, batch=-1)

#  Valid formats are ('torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs', 'paddle', 'ncnn')
model.export(format="saved_model", keras=True)
