from pathlib import Path

import pandas as pd
from PIL import Image


class Irma:
    """This is the IRMA data set.

    Paper
    -----
    Lehmann, T. M., Schubert, H., Keysers, D., Kohnen, M., & Wein, B. B.,
    The IRMA code for unique classification of medical images,
    In, Medical Imaging 2003: PACS and Integrated Medical Information Systems:
    Design and Evaluation (pp. 440–451) (2003). : International Society for Optics and Photonics.

    https://www.kaggle.com/raddar/irma-xray-dataset
    """

    def __init__(self, root, *args, **kwargs):
        self.data_dir = Path(root)

    def load(self):
        self.train_labels_path = self.data_dir / "ImageCLEFmed2009_train_codes.02.csv"
        self.train_images_path = self.data_dir / "ImageCLEFmed2009_train.02/ImageCLEFmed2009_train.02"

        df = pd.read_csv(self.train_labels_path, delimiter=";")
        df.loc[:, "Path"] = df["image_id"].apply(self._get_image_path)
        df.loc[:, "Technical Code"] = df["irma_code"].apply(self._get_technical_code)
        df.loc[:, "Imaging Modality"] = df["Technical Code"].apply(self._get_imaging_modality)
        df.loc[:, "Directional Code"] = df["irma_code"].apply(self._get_directional_code)
        df.loc[:, "Imaging Orientation"] = df["Directional Code"].apply(self._get_imaging_orientation)
        df.loc[:, "Anatomical Code"] = df["irma_code"].apply(self._get_anatomical_code)
        df.loc[:, "Body Region"] = df["Anatomical Code"].apply(self._get_body_region)
        df.loc[:, "Biological Code"] = df["irma_code"].apply(self._get_biological_code)
        df.loc[:, "Body Region Label"] = df["Anatomical Code"].apply(self._get_body_region_label)
        self.df = df[["image_id", "irma_code", "Path", "Technical Code", "Directional Code", "Anatomical Code",
                      "Biological Code", "Imaging Modality", "Body Region", "Body Region Label"]]

    def load_image(self, path: str) -> Image:
        """Cache and load an image."""
        return Image.open(path).convert("RGB")
        #return Image.open(path).convert("L")

    def _get_image_path(self, image_id: str) -> str:
        return self.train_images_path / f"{image_id}.png"

    def _get_technical_code(self, irma_code: str) -> str:
        return irma_code.split("-")[0]

    def _get_imaging_modality(self, technical_code: str):
        first, second, third, fourth = technical_code
        first_categories = {"0": "unspecified",
                            "1": "x-ray",
                            "2": "sonography",
                            "3": "magnetic resonance measurements",
                            "4": "nuclear medicine",
                            "5": "optical imaging",
                            "6": "biophysical procedure",
                            "7": "others",
                            "8": "secondary digitalization"}
        if first in first_categories:
            return first_categories[first]
        return technical_code

    def _get_directional_code(self, irma_code: str) -> str:
        return irma_code.split("-")[1]

    def _get_imaging_orientation(self, directional_code: str) -> str:
        first, second, third = directional_code
        result = directional_code
        if first == 0:
            return "unspecified"
        elif first == 1:
            if second == 1:
                return "posteroanterior"
            elif second == 2:
                return "anteroposterior"
        elif first == 2:
            if second == 1:
                return "lateral, right-left"
            elif second == 2:
                return "lateral, left-right"
        return result

    def _get_anatomical_code(self, irma_code: str) -> str:
        return irma_code.split("-")[2]

    def _get_body_region(self, anatomical_code: str) -> str:
        first, second, third = anatomical_code
        first_categories = {
            "1": "whole body",
            "2": "cranium",
            "3": "spine",
            "4": "upper extremity/arm",
            "5": "chest",
            "6": "breast",
            "7": "abdomen",
            "8": "pelvis",
            "9": "lower extremity"
        }
        if first in first_categories:
            if second == "5":
                chest_categories = {
                    "0": "chest",
                    "1": "chest/bones",
                    "2": "chest/lung",
                    "3": "chest/hilum",
                    "4": "chest/mediastinum",
                    "5": "chest/heart",
                    "6": "chest/diaphragm"
                }
                return chest_categories[second]
            return first_categories[first]
        return anatomical_code

    def _get_body_region_label(self, anatomical_code: str) -> str:
        first, second, third = anatomical_code
        return int(first)-1

    def _get_biological_code(self, irma_code: str) -> str:
        return irma_code.split("-")[3]