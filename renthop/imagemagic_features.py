import subprocess
import yaml
import json
from tqdm import tqdm
from time import time
from glob import glob
import os

import traceback

hist = 'Histogram:'
colormap = 'Colormap:'


from datetime import datetime

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, datetime):
        serial = obj.isoformat()
        return serial
    #return '%NOT_SERIALIZABLE%'
    raise TypeError("Type not serializable")

def process_image(file_path, result_file):
    try:
        data = imagemagick_features(file_path)
        if data is None:
            return

        image_id = os.path.basename(file_path)
        result_file.write(image_id)
        result_file.write('\t')
        result_file.write(json.dumps(data, default=json_serial))
        result_file.write('\n')
        result_file.flush()
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        return None

def imagemagick_features(file_path):
    try:
        output = subprocess.check_output(['identify', '-verbose', '-moments', file_path])
        return try_process_output(output)
    except Exception as e:
        print(str(e))
        traceback.print_exc()
        return None

def try_process_output(output):
    output = output.decode('utf8')
    output_res = output.split('\n')[1:-4]
    
    form = [s[2:] for s in output_res if s and (not 'comment:' in s)]

    remove_entry(hist, form)
    remove_entry(colormap, form)
    remove_entry('unknown[2,0]:', form)
    

    form = '\n'.join(form)
    form = yaml.load(form)

    del form['Artifacts']
    del form['Class']
    del form['Dispose']
    del form['Endianess']
    del form['Compression']
    del form['Units']
    del form['Properties']['date:create']
    form['Properties']['date:modify'] = str(form['Properties']['date:modify'])
    return form

def remove_entry(name, form):
    if name not in form:
        return
    idx = form.index(name)
    end_idx = idx + 1

    while end_idx < len(form):
        if not form[end_idx].startswith(' '):
            break
        end_idx = end_idx + 1

    del form[idx:end_idx]


pbar = tqdm(total=699162)

result = open('imagemagick_features.txt', 'w')


for root, dirs, files in os.walk('data/images/'):
    for f in files:
        if not f.endswith('.jpg'):
            continue
        img_file = os.path.join(root, f)
        process_image(img_file, result)
        pbar.update(1)
    
result.close()
