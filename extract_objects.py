import os
import cv2
import numpy as np
from PIL import Image
from PIL import ImageOps
#from yolo_segme import yolo_txt_box

RES_DIR = ''

def yolo_txt_box(yolo_coords, img_width, img_height):
    x_center, y_center, width, height = yolo_coords
    x1 = int((x_center - width/2) * img_width)
    y1 = int((y_center - height/2) * img_height)
    x2 = int((x_center + width/2) * img_width)
    y2 = int((y_center + height/2) * img_height)

    return [x1, y1, x2, y2]

def save_img(img_dir):
    img_folder_name = os.path.basename(img_dir)
    labels_folder = os.path.join(os.path.dirname(img_dir), f'label files ({img_folder_name})') 
    output_dir = os.path.join(RES_DIR, f'{os.path.basename(os.path.dirname(img_dir))}')
    os.makedirs(output_dir, exist_ok=True)

    print(f'Обработка папки: {os.path.basename(os.path.dirname(img_dir))}')

    for img in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img)
        img_name = os.path.splitext(img)[0]
        label_path = os.path.join(labels_folder, f'{img_name}.txt')
        image = ImageOps.exif_transpose(Image.open(img_path))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = image.shape

        if not os.path.exists(label_path):
            continue

        with open(label_path) as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            cls_xyxy = list(map(float, line.strip().split()))
            x1, y1, x2, y2 = yolo_txt_box(cls_xyxy[1::], w, h)
            image_copy = image.copy()
            cut = image_copy[y1:y2, x1:x2]

            cut_filename = os.path.join(output_dir, f'{img_name}_class_{int(cls_xyxy[0])}_{idx}.jpg')
            _, buffer = cv2.imencode('.jpg', cut)
            with open(cut_filename, 'wb') as f:
                f.write(buffer)

    return
        
def find_img_folder(dir):
    for root, _, files in os.walk(dir):
        img = any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files)

        if img:
            save_img(root)
    
    return 
    
if __name__ == '__main__':
    print('Start')
    find_img_folder('')