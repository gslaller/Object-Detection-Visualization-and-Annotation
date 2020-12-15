from flask import Flask, request, send_from_directory, jsonify
import os
from inference import Inference

app = Flask(__name__)

PATH_TO_IMAGES = "./temp"
INFERENCE = None

if not os.path.exists(PATH_TO_IMAGES):
  os.mkdir(PATH_TO_IMAGES)

@app.route("/path_exists", methods=['POST'])
def path_exists():
  path = request.get_json(force = True)['path']
  if(os.path.exists(path)):
    return "true"
  else:
    return "false"

@app.route("/model_initialize", methods=['POST'])
def model_initialize():
  global INFERENCE
  json = request.get_json(force=True)
  weight_path = json['weight_path']
  project_path = json['project_path']
  try:
    INFERENCE = Inference(weight_path, project_path)
    return "true"
  except:
    return "false"

@app.route("/list_images")
def list_images():
  return jsonify(os.listdir(PATH_TO_IMAGES))

@app.route("/upload_image", methods=['POST'])
def upload():
  upload_files = request.files.getlist("files[]")
  for file in upload_files:
    file.save(f'{PATH_TO_IMAGES}/{file.filename}')
  return "true"

@app.route("/infer", methods=['POST'])
def infer():
  json = request.get_json(force = True)
  img_path = os.path.join(PATH_TO_IMAGES,json['img_path'])
  acc_threshold = float(json['acc_thres'])
  iou_threshold = float(json['iou_thres'])
  if not os.path.exists(img_path):
    return jsonify({})
  
  data = INFERENCE.forward(img_path, acc_threshold, iou_threshold)
  for key in data.keys():
    data[key] = data[key].tolist()
    
  return jsonify(data) 

@app.route("/image/<path:path>")
def get_image(path):
  return send_from_directory(PATH_TO_IMAGES, path)

@app.route("/")
def root():
  return send_from_directory('.', 'index.html')

if __name__ == "__main__":
  app.run(debug=True)
