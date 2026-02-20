import os
import cv2
import numpy as np
from src.datasets.Dataset_Base import Dataset_Base


class ISIC2018_Dataset(Dataset_Base):
    """Dataset loader for ISIC 2018 skin lesion segmentation.
    Expects directory structure:
        img_path/  -> ISIC_0000000.jpg, ISIC_0000001.jpg, ...
        label_path/ -> ISIC_0000000_segmentation.png, ISIC_0000001_segmentation.png, ...
    """
    def __init__(self, input_channels=3, resize=True):
        self.slice = 0  # Set non-None so agent uses 2D code path
        self.input_channels = input_channels
        super().__init__(resize)

    def getFilesInPath(self, path):
        """Get files in path, organized by patient ID.
        Args:
            path: directory containing images or masks
        Returns:
            dic: {patient_id: {0: (filename, patient_id, 0)}}
        """
        dir_files = sorted(os.listdir(path))
        dic = {}
        for f in dir_files:
            if f.startswith('.'):
                continue
            # Extract patient ID (e.g., ISIC_0000000 from ISIC_0000000.jpg or ISIC_0000000_segmentation.png)
            patient_id = f.split('_segmentation')[0]
            patient_id = os.path.splitext(patient_id)[0]
            if patient_id not in dic:
                dic[patient_id] = {}
            dic[patient_id][0] = (f, patient_id, 0)
        return dic

    def __getitem__(self, idx):
        """Load and preprocess an image-mask pair.
        Args:
            idx: index into the dataset
        Returns:
            (id, image, mask) where image is [H, W, C] and mask is [H, W, 1]
        """
        img_data = self.data.get_data(key=self.images_list[idx])
        if not img_data:
            img_name, p_id, _ = self.images_list[idx]
            label_name, _, _ = self.labels_list[idx]

            # Load image
            img_path = os.path.join(self.images_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load mask
            label_path = os.path.join(self.labels_path, label_name)
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise FileNotFoundError(f"Mask not found: {label_path}")

            # Resize to target size
            if self.resize and hasattr(self, 'size'):
                img = cv2.resize(img, dsize=self.size, interpolation=cv2.INTER_CUBIC)
                label = cv2.resize(label, dsize=self.size, interpolation=cv2.INTER_NEAREST)

            # Normalize image to [0, 1]
            img = img.astype(np.float32) / 255.0

            # Handle input channels
            if self.input_channels == 1:
                img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
            # else keep RGB [H, W, 3]

            # Binarize mask and add channel dim
            label = label.astype(np.float32)
            label[label > 0] = 1.0
            label = np.expand_dims(label, axis=-1)  # [H, W, 1]

            img_id = "_" + str(p_id) + "_0"
            self.data.set_data(key=self.images_list[idx], data=(img_id, img, label))
            img_data = self.data.get_data(key=self.images_list[idx])

        img_id, img, label = img_data

        # Limit to configured channels
        if hasattr(self, 'exp') and self.exp is not None:
            input_ch = self.exp.get_from_config('input_channels')
            output_ch = self.exp.get_from_config('output_channels')
            if input_ch is not None:
                img = img[..., :input_ch]
            if output_ch is not None:
                label = label[..., :output_ch]

        return (img_id, img, label)
