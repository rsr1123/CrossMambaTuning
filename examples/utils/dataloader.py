from glob import glob

from torch.utils.data import Dataset
from PIL import Image

class MSCOCO(Dataset):
    def __init__(self, root, transform, img_list=None):
        assert root[-1] == '/', "root to COCO dataset should end with \'/\', not {}.".format(
            root)

        if img_list:
            self.image_paths = []
            with open(img_list, 'r') as r:
                lines = r.read().splitlines()
                for line in lines:
                    self.image_paths.append(root + line)
        else:
            self.image_paths = sorted(glob(root + "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)

from pycocotools.coco import COCO
from detectron2.structures import Boxes, Instances
import os
import torch
from torchvision import transforms

class MSCOCO_t(Dataset):
    def __init__(self, root, transform, annotation_file, img_list=None):

        super().__init__()
        assert root.endswith('/'), f"root to COCO dataset should end with '/', not {root}."

        self.transform = transform

        self.coco = COCO(annotation_file)
        self.filename_to_id = {img['file_name']: img['id'] for img in self.coco.dataset['images']}

        if img_list:
            with open(img_list, 'r') as r:
                lines = r.read().splitlines()
            self.image_paths = [os.path.join(root, line) for line in lines if line in self.filename_to_id]
        else:
            all_paths = sorted(glob(os.path.join(root, "*.jpg")))
            self.image_paths = [p for p in all_paths if os.path.basename(p) in self.filename_to_id]

        print(f"Initialized MSCOCO dataset. Found {len(self.image_paths)} images with annotations.")

    def __getitem__(self, index):

        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')

        file_name = os.path.basename(img_path)
        image_id = self.filename_to_id[file_name]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        classes = []
        for ann in anns:
            if ann.get('iscrowd', 0) == 0:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                classes.append(ann['category_id'])

            return self.__getitem__((index + 1) % len(self))

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        H, W = img_tensor.shape[1:]
        instances = Instances(image_size=(H, W))
        instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

        annotation_dict = {
            "height": H,
            "width": W,
            "image_id": image_id,
            "instances": instances
        }

        return img_tensor, annotation_dict

    def __len__(self):
        return len(self.image_paths)

class Kodak(Dataset):

    def __init__(self, root, transform):
        if not root.endswith('/') and not root.endswith('\\'):
            root = root + '/'
        self.image_paths = sorted(
            glob(root + "*.png") + glob(root + "*.jpg") + glob(root + "*.jpeg")
        )
        assert len(self.image_paths) > 0, f"鏈湪 {root} 鎵惧埌浠讳綍鍥惧儚鏂囦欢"
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.image_paths)
class COCO_test(Dataset):
    def __init__(self, root, transform):

        assert root[-1] == '/', "root to Kodak dataset should end with \'/\', not {}.".format(
            root)

        self.image_paths = sorted(glob(root + "*.jpg"))
        self.transform = transform

    def __getitem__(self, index):

        img_path = self.image_paths[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.image_paths)


