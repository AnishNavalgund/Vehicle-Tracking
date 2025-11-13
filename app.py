import gradio as gr
from ultralytics import YOLO

# Load model 
model = YOLO("yolo11n.pt")  

def detect(image):
    results = model(image)[0]
    annotated = results.plot()  # numpy image with bounding boxes
    return annotated

demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="YOLOv11 Object Detection"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
