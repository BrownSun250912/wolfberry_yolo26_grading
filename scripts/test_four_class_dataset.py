import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from build_four_class_dataset import assign_instances, read_polygons


class DatasetTests(unittest.TestCase):
    def test_matching_is_spatial_not_instance_suffix_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            folder=root/'outputs/features_frozen_v1/train'
            folder.mkdir(parents=True)
            a=np.zeros((50,50),np.uint8); a[5:15,5:15]=1
            b=np.zeros((50,50),np.uint8); b[30:40,30:40]=1
            rows=[]
            for i,mask in enumerate([b,a]):
                name=f'{i}.png'; cv2.imwrite(str(folder/name),mask*255)
                rows.append({'instance_id':f'fruit__i{i+1:03d}','split':'train',
                             'raw_mask_path':name,'roi_x1':0,'roi_y1':0,'roi_x2':50,'roi_y2':50})
            cfg={'minimum_iou':.5,'minimum_ambiguity_margin':.1}
            assigned,safe=assign_instances(rows,[{'source_row':1},{'source_row':2}],[a,b],root,cfg)
            self.assertTrue(safe)
            self.assertEqual(assigned[0]['instance_id'],'fruit__i002')
            self.assertEqual(assigned[1]['instance_id'],'fruit__i001')
            _,safe=assign_instances(rows[:1],[{'source_row':1},{'source_row':2}],[a,b],root,cfg)
            self.assertFalse(safe)
            _,safe=assign_instances(rows,[{'source_row':1},{'source_row':2}],[a,a],root,cfg)
            self.assertFalse(safe)

    def test_exact_duplicate_is_audited_and_invalid_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for folder in ('labels','labelme_json'):
                (root/folder/'train').mkdir(parents=True)
            line='0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4'
            (root/'labels/train/x.txt').write_text(line+'\n'+line+'\n')
            shape={'label':'wolfberry','shape_type':'polygon','points':[[10,10],[40,10],[40,40],[10,40]]}
            doc={'imageWidth':100,'imageHeight':100,'shapes':[shape,shape]}
            jp=root/'labelme_json/train/x.json'; jp.write_text(json.dumps(doc))
            rec={'new_stem':'x','split':'train','image_width':100,'image_height':100}
            cfg={'minimum_polygon_area_px2':16,'json_txt_pixel_tolerance':.001}
            polys,masks,dups,_=read_polygons(root,Path('.'),rec,cfg)
            self.assertEqual((len(polys),len(masks),len(dups)),(1,1,1))
            self.assertEqual(dups[0]['duplicate_source_row'],2)
            doc['shapes'][0]['points'][0][0]=11
            jp.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,'json_txt_geometry_mismatch'):
                read_polygons(root,Path('.'),rec,cfg)


if __name__=='__main__':
    unittest.main()
