import sys

sys.path.append("../src")

from typing import List
from typing import Tuple

import tempfile
from pathlib import Path

import torch
from IPython.display import display
from IPython.display import Markdown

from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import textwrap

from health_multimodal.common.visualization import plot_phrase_grounding_similarity_map
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.vlp import ImageTextInferenceEngine

import torch.nn.functional as F
from scipy import ndimage
from health_multimodal.image.model.model import BaseImageModel
from health_multimodal.text import TextInferenceEngine
from health_multimodal.image import ImageInferenceEngine
from health_multimodal.image.data.transforms import infer_resize_params
from typing import Callable, List, Optional, Union
from math import ceil, floor
from health_multimodal.image.data.io import load_image

from model import ImageTextModel

train_dir_name = "../../../train"
df = pd.read_csv("../../../label_1024_split.csv")


# Load BioViL Model
text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_inference = get_image_inference(ImageModelType.BIOVIL_T)

# BioViL = ImageTextModel(image_inference, text_inference, 1024, 1024)
print(len(df))
df_train = df.query('split == "train"')

width, height = df_train.iloc[0]["image_width"], df_train.iloc[0]["image_height"]
model = ImageTextModel(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
    width=width,
    height=height,
)


class CustomDataset2(Dataset):
    def __init__(self, dataframe, device, transform):
        self.dataframe = dataframe
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = Path(f"{train_dir_name}/{row.dicom_id}.jpg")
        # image = Image.open(image_path).convert('RGB')
        # image = torch.tensor(np.array(image))
        text_prompt = row.label_text
        ground_truth_boxes = torch.tensor([row.x, row.y, row.w, row.h])
        image = load_image(image_path)
        transformed_image = self.transform(image)

        return transformed_image, text_prompt, ground_truth_boxes


batch_size = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = CustomDataset2(df_train, device, image_inference.transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# train_loader = DataLoader(MyDataset(), batch_size=32, shuffle=True)

import torch.nn as nn

criterion = nn.MSELoss()
optimizer = optim.Adam(model.text_inference_engine.model.parameters(), lr=0.001)

n_epochs = 10
# model.to(device)

for epoch in range(n_epochs):
    # model.train()

    for batch_idx, (images, text_prompt, ground_truth_boxes) in enumerate(train_loader):
        # predicted_boxes = model(image_path, text_prompt)
        # print(predicted_boxes.cpu().detach().numpy())
        # print(ground_truth_boxes.cpu().detach().numpy())
        # loss = criterion(predicted_boxes, ground_truth_boxes.view(1, -1).float())
        # print(ground_truth_boxes)
        loss = 0

        similarity_map = model.get_similarity_maps_from_raw_data(
            images=images,
            query_text=text_prompt,
            interpolation="bilinear",
        ).clip(0)
        assert similarity_map.shape[1] == 1024
        assert similarity_map.shape[2] == 1024

        tmp_batch_size = images.shape[0]

        for i in range(tmp_batch_size):
            row_x, row_y, row_w, row_h = (ground_truth_boxes[i]).detach().int()

            # Calculate the sum within the box
            sum_val = torch.sum(
                similarity_map[i][row_x : row_x + row_w, row_y : row_y + row_h]
            )
            loss -= sum_val / torch.sum(similarity_map[i]) / tmp_batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 1 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}"
            )
