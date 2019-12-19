import numpy as np
import cv2, os, random

from os.path import join


class WiderFaceDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self._train_ls = self.load_file("wider_face_train_bbx_gt.txt")
        self._val_ls = self.load_file("wider_face_val_bbx_gt.txt")
        self._test_ls = self.load_file("wider_face_test_filelist.txt")

    def load_file(self, file_name):
        result = []

        file_path = join(self.data_dir, "wider_face_split", file_name)
        if file_name == "wider_face_test_filelist.txt":
            return [line.strip() for line in open(file_path, "r").readlines()]

        with open(file_path, "r") as f:
            status = 0
            for line in f.readlines():
                line = line.strip()
                if status == 0:
                    record = [line, []]
                    status = 1
                    continue

                if status == 1:
                    count = int(line)
                    status = 2
                    continue

                if status == 2:
                    record[1].append([int(s) for s in line.split(" ")])
                    count -= 1
                    if count <= 0:
                        result.append(record)
                        status = 0
                    continue

        return result

    def load_image(self, image_path):
        img = cv2.imread(join(self.data_dir, image_path))
        return img / 255.

    def train_data(self):
        random.shuffle(self._train_ls)
        for image_name, data in self._train_ls:
            image_path = join("WIDER_train", "images", image_name)
            img = self.load_image(image_path)
            roi = np.array(data)[:, :4].astype(np.float)
            yield img, roi

    def val_data(self):
        for image_name, data in self._val_ls:
            image_path = join("WIDER_val", "images", image_name)
            img = self.load_image(image_path)
            roi = np.array(data)[:, :4].astype(np.float)
            yield img, roi

    def test_data(self):
        for image_name, data in self._test_ls:
            image_path = join("WIDER_test", "images", image_name)
            img = self.load_image(image_path)
            yield img


if __name__ == '__main__':
    dataset = WiderFaceDataset("/home/killf/data/数据集/wider_face")
    for i in dataset.train_data():
        pass
    print(dataset)
