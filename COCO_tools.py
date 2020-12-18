import os
import json
from PIL import Image
import re
import time
import yaml

"""
images: file_name, height, width, id, license: 1
annotations: bbox, category_name
"""

class COCO_Annotation():
    def __init__(self, directory_path, project_path):
        self.dirname = directory_path
        self.project_path = project_path
        self.image_paths = self.get_images_path(self.dirname)
        for image_path in self.image_paths:
            _ = self._read_json(image_path)
    
    def get_images_path(self, path):
        return self._list_dir(self.dirname, suf=[".png", ".jpg"])
     
    def _list_dir(self, path, pre=[""], suf=[""]):
        lsdir = os.listdir(path)
        check_pre = lambda ele: any([ele[:len(sub_ele)] == sub_ele for sub_ele in pre])
        check_suf = lambda ele: any([ele[-len(sub_ele):] == sub_ele for sub_ele in suf])
        lsdir = [ele for ele in lsdir if check_pre(ele) and check_suf(ele)]
        return lsdir
    
    def _create_empty_json(self, image_path):
        full_image_path = os.path.join(self.dirname, image_path)
        width, height = Image.open(full_image_path).size
        empty_json = {
            "file_name": image_path,
            "height": height,
            "width": width,
            "annotations": []
        }
        return empty_json
    
    def update_annotations(self, image_path, updated_annotation):
        data = self._read_json(image_path)
        data['annotations'] = updated_annotation
        self._write_json(image_path, data)
        
    
    def del_annotation(self, image_path, uid):
        old_annotations = self.get_annotation(image_path)
        new_annotations = [ele for ele in old_annotations if ele['uid'] != uid]
        self.update_annotations(image_path, new_annotations)
    
    def append_annotation(self, image_path, new_annotation):
        #new_annotation has two keys: bbox, category_name
        #    and the uid is created on the fly.
        
        old_annotations = self.get_annotation(image_path)
        
        if len(old_annotations) == 0:
            new_uid = 0
        else:
            new_uid = old_annotations[-1]["uid"]+1
        
        self.append_category(new_annotation['category_name'])
        
        new_annotation['uid'] = new_uid
        new_annotations = old_annotations + [new_annotation]
        self.update_annotations(image_path, new_annotations) 
    
    def get_annotation(self, image_path):
        annotations = self._read_json(image_path)['annotations']
        return annotations

    def _load_yaml(self, path):
        with open(path, "r") as file:
            return yaml.safe_load(file)
    
    def _write_yaml(self, path, data):
        with open(path, "w+") as file:
            yaml.safe_dump(data, file )
    
    def get_categories(self):
        data = self._load_yaml(self.project_path)
        categories = data['obj_list']
        return categories
    
    def append_category(self, new_cat):
        #append new_cat if it does not exist
        current_cat = self.get_categories()
        if new_cat not in current_cat:
            new_cat = current_cat+[new_cat]
            self._update_categories(current_cat)

    def _update_categories(self, updated_categories):
        data = self._load_yaml(self.project_path)
        data['obj_list'] = updated_categories
        self._write_yaml(self.project_path, data)
    
    def _image_2_json_path(self, image_path):
        basename = os.path.splitext(image_path)[0]
        return "."+basename+".json"

    def _write_json(self, image_path, data_json):
        json_path = self._image_2_json_path(image_path)
        json_path = os.path.join(self.dirname, json_path)
        with open(json_path, "w+") as file:
            json.dump(data_json, file)
    
    def _read_json(self, image_path):
        json_path = self._image_2_json_path(image_path)
        json_path = os.path.join(self.dirname, json_path)
        if not os.path.exists(json_path):
            empty_json = self._create_empty_json(image_path)
            self._write_json(image_path, empty_json)
            return empty_json
         
        with open(json_path, "r") as file:
            return json.load(file)

class Merge_JSON_Files():
    def __init__(self, directory, project_path, new_file_path, new_project_path):
        self.dirname = directory
        self.project_path = project_path
        self.new_project_path = new_project_path
        self.main_json = self.get_dummy_json()
        
        json_files = [os.path.join(self.dirname, e) for e in os.listdir(self.dirname) if e[:1] == "." and e[-5:] == ".json"]
        
        self.categories = self.load_yaml(self.project_path)['obj_list']
        self.images = []
        self.annotations = [] 
        
        self.image_id = 0
        self.annotation_id = 0
        
        for json_file in json_files:
            self.process_file(json_file)
        
        self.create_new_yaml()
        self.postprocess_categories()
        
        with open(new_file_path, "w+") as file:
            json.dump(self.main_json, file)
        
    
    def get_dummy_json(self):
        return {
            'info': time.ctime(),
            'lincense': '__author__',
            'categories': [],
            'images': [],
            'annotations': []
        }
    
    def append_image(self, data):
        self.main_json['images'].append({
            **data, 
            'id': self.image_id,
            'license': 1,
        })
        self.image_id += 1
        return self.image_id
    
    def append_annotation(self, data, image_id):
        category_name = data['category_name']
        bbox = data['bbox']
        category_id = self.append_categories(category_name)
        self.main_json['annotations'].append({
            'bbox': bbox,
            'category_id': category_id,
            'id':  self.annotation_id,
            'image_id': image_id,
        })
        self.annotation_id += 1
        
        
    def process_file(self, path):
        with open(path, "r") as file:
            json_data = json.load(file)
            """
                so json_data = [
                    file_name:
                    height:
                    width:
                    annotations: [category_name, bbox]
                ]
            """
            image_id = self.append_image({k:v for k,v in json_data.items() if k in ['file_name', 'height', 'width']})
            
            for annotation in json_data['annotations']:
                self.append_annotation(annotation, image_id)

    def append_categories(self, category_name):
        
        if category_name not in self.categories:
            self.categories.append(category_name)
        #idx has to be +1
        idx = self.categories.index(category_name)
        return idx + 1

    def load_yaml(self, path):
        with open(path, "r") as file:
            return yaml.safe_load(file)
    
    def write_yaml(self, path, data):
        with open(path, "w+") as file:
            yaml.safe_dump(data, file )

    def create_new_yaml(self):
        data = self.load_yaml(self.project_path)
        data['obj_list'] = self.categories
        self.write_yaml(self.new_project_path, data)

    def postprocess_categories(self):        
        res = []
        for idx, cat in enumerate(self.categories):
            res.append({
                'id': idx+1,
                'name': cat,
                'supercategory': "None"
            })
        self.categories = res




if __name__ == "__main__":
    project_path = "./projects/coco.yml"
    directory_path = "./temp"
    ca = COCO_Annotation(directory_path, project_path)
    import random
    
    image_files = [e for e in os.listdir(directory_path) if e[-5:] != ".json"]
    
    r_path = lambda : random.choice(image_files)
    
    categories = lambda : str(random.randint(0,20))
    
    
    def random_data():
        return {
            'category_name': categories(),
            'bbox': [random.randint(0,100) for _ in range(4)]
        }
    
    for _ in range(50):
        ca.append_annotation(r_path(), random_data())
    Merge_JSON_Files(directory_path, project_path, "something_1.json", "./projects/coco_1.yml")
    print("done")