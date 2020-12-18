from flask import Flask, request, send_from_directory, jsonify
import os
from inference import Inference
from COCO_tools import COCO_Annotation, Merge_JSON_Files
import yaml

app = Flask(__name__)

PATH_TO_IMAGES = "./temp"
PATH_TO_WEIGHT = "./weights/efficientdet-d0.pth"
PATH_TO_PROJECT = "./projects/custom.yml"
COCO = None
INFERENCE = None

if not os.path.exists(PATH_TO_IMAGES):
  os.mkdir(PATH_TO_IMAGES)

@app.route("/check_dir", methods=['POST'])
def check_dir():
  path = request.get_json(force = True)['past']
  if os.path.exists(path):
    retos.listdir(directory_path)
    return "true"
  else:
    return "false"

@app.route("/initalize_all", methods=['POST'])
def initalize_all():
  global COCO, PATH_TO_IMAGES, PATH_TO_PROJECT, PATH_TO_PROJECT, INFERENCE
  json = request.get_json(force = True)
  PATH_TO_IMAGES = json['directory_path']
  PATH_TO_PROJECT = json['project_path']
  PATH_TO_WEIGHT = json['weight_path']
  try:
    COCO = COCO_Annotation(PATH_TO_IMAGES, PATH_TO_PROJECT)
    INFERENCE = Inference(PATH_TO_WEIGHT, PATH_TO_PROJECT)
    return "true"
  except:
    return "false"

@app.route("/merge_all_json", methods=['POST'])
def merge_all_json():
  #mergedFile path
  #new_project_file_path
  json = request.get_json(force = True)
  new_annotation_file_path = json['annotation_file_path']
  new_project_file_path = json['project_file_path']
  try:
    Merge_JSON_Files(PATH_TO_IMAGES, PATH_TO_PROJECT, new_annotation_file_path, new_project_file_path)
    return "true"
  except:
    return "false" 

@app.route("/get_annotation", methods=['POST'])
def get_annotation():
  json = request.get_json(force = True)
  file_name = json['file_name']
  data = COCO.get_annotation(file_name)
  return jsonify(data)



@app.route("/append_annotation", methods=['POST'])
def append_annotation():
  json = request.get_json(force = True)
  file_name = json['file_name']
  category_name = json['category_name']
  bbox = json['bbox']
  print(file_name, category_name, bbox)
  annotation = {
    "category_name": category_name,
    "bbox": bbox,
  }
  COCO.append_annotation(file_name, annotation)
  return "true" 

@app.route("/remove_annotation", methods=['POST'])
def remove_annotation():
  json = request.get_json(force = True)
  uid = json['uid']
  file_name = json['file_name']
  COCO.del_annotation(file_name, uid)
  return "true"

@app.route("/list_images")
def list_images():
  ls = COCO.get_images_path(PATH_TO_IMAGES)
  return jsonify(ls)

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

@app.route("/obj_list")
def obj_list():
  try:
    with open(PATH_TO_PROJECT, "r") as file:
      data = yaml.safe_load(file)
      return jsonify(data['obj_list'])
  except:
    return jsonify([])

@app.route("/")
def root():
  return send_from_directory('.', 'index.html')

if __name__ == "__main__":
  app.run(debug=True)
