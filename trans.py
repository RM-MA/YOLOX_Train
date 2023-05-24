import json
import sys



a = json.load(open(sys.argv[1]))


a_new = {key:[] for key in a.keys()}


a_new['info'] = a['info']
a_new['licenses'] = a['licenses']
a_new['images'] = a['images']

for i,c in enumerate(a['categories']):
  # print(a['categories'])
  if c['id'] != 0:
    a_new['categories'].append({
      'id': c['id'] - 1,
      'name': c['name']
    })
    
    
for an in a['annotations']:
  ks = []
  for i in range(5):
    ks.append(an['segmentation'][0][i*2])
    ks.append(an['segmentation'][0][i*2 + 1])
    ks.append(2)
  a_new['annotations'].append({
    'id': an['id'],
    'image_id': an['image_id'],
    'category_id': an['category_id'] - 1,
    'area': an['area'],
    'segmentation': [an['segmentation'][0][:10]],
    'bbox': an['bbox'],
    'iscrowd': 0,
    'num_keypoints': 5,
    'keypoints': ks
  })
  
  
  
json.dump(a_new, open(sys.argv[1], "w"))