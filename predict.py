import json
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from cog import BaseModel, BasePredictor, Input, Path
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from models.anchornet import AnchorNet
from models.densenet import densenet201
from utils import IOU


class Output(BaseModel):
    plot: Path = None
    Json: str = None


class Predictor(BasePredictor):
    def setup(self):
        self.class_names = get_imagenet_classnames()
        self.anchornet, self.down_net, self.resized_down_net = load_models()
        self.patch_size = 95
        self.num_classes = 1000
        self.trans1 = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
        self.trans2 = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        input_image: Path = Input(description="Image to be classified"),
        output_format: str = Input(
            description="Final-probs-plot: Plots the final prediction of top 10 classes\n"
                        "Json: Get the final prediction of top 10 classes in json format\n"
                        "Predictions-plot: Plot intermediate predictions of patches\n"
                        "CAM-plot: Plot the intermediate CAM map from AnchorNet",
            default="Predictions-plot",
            choices=["Final-probs-plot", "Json", "Predictions-plot", "CAM-plot"],
        ),
        iou_thresholds: str = Input(
            description="Comma separated list of floats where the ith float determines the allowed IOU to a next (i+1) patch proposal (6 max). Leave empty for a sinle patch proposal",
            default=""
        ),
    ) -> Output:
        iou_thresholds = str(iou_thresholds.strip())
        iou_thresholds = [float(x) for x in iou_thresholds.split(",")] if iou_thresholds else []
        with torch.no_grad():
            # Load image
            raw_image = Image.open(str(input_image)).convert("RGB")
            raw_image = self.trans1(raw_image)
            image = self.trans2(raw_image.copy()).unsqueeze(0).cuda()

            # Get proposed patches
            feas, fc_weights, anchornet_logits = self.anchornet(image)
            anchornet_class = anchornet_logits.argmax(dim=1)
            cam_weight = torch.index_select(fc_weights, 0, anchornet_class)
            cam_weight = cam_weight.unsqueeze(-1).unsqueeze(-1)
            cam = torch.mean(feas * cam_weight, dim=1)
            patches, boxes = crop_patches(image, cam, patch_size=self.patch_size,
                                          stride=8, iou_thresholds=iou_thresholds,
                                          )
            n_patches = len(patches)
            # Classify patches
            patch_logits = self.down_net(patches)

            # Classify resized image
            resized_inputs = F.interpolate(image, size=[self.patch_size, self.patch_size],
                                           mode="bicubic", align_corners=True,)
            resized_logits = self.resized_down_net(resized_inputs)[0]

            # Parse results
            aggregated_logits = resized_logits + patch_logits.sum(0)

            results_dict = OrderedDict({
                "AnchorNet": self.get_pred_string(anchornet_logits[0]),
                "Resized_input": self.get_pred_string(anchornet_logits[0])
            })

            for i in range(n_patches):
                results_dict[f"Patch #{i}"] = self.get_pred_string(patch_logits[i])

            results_dict["aggregated"] = self.get_pred_string(aggregated_logits)
            results_dict = dict(results_dict)

            # Manage outputs:
            final_probs = F.softmax(aggregated_logits, dim=0).cpu().numpy()
            topk = np.argsort(final_probs)[::-1][:10]
            output_path = "output.png"
            if output_format == "Json":
                results_dict = {self.class_names[k]: f"{final_probs[k]:.3f}" for k in topk}
                return Output(Json=json.dumps(results_dict))
            elif output_format == "Final-probs-plot":
                plot_topk_classes([self.class_names[k] for k in topk], final_probs[topk], output_path)
            elif output_format == "CAM-plot":
                visualize_CAM(cam[0].cpu().numpy(), self.class_names[anchornet_class.item()], output_path)
            else:
                visualize_predictions(raw_image, boxes, n_patches, results_dict, output_path)
            return Output(plot=Path(output_path))

    def get_pred_string(self, logits):
        probs = F.softmax(logits, dim=0)
        confidence, max_class = probs.max(dim=0)
        return f"{self.class_names[max_class.item()]}: {confidence.item():.3f}"


def localize_bbox(cam, patch_size, stride, iou_thresholds):
    pos_list = pos_lists()
    value = cam.flatten()
    argmaxx = value.argsort(descending=True)
    try_n_patches = len(iou_thresholds) + 1
    final_bbox = []
    count = 0
    for i in argmaxx:
        pt1 = (pos_list[i][0] * stride, pos_list[i][1] * stride)
        bbox = (pt1[0], pt1[1], pt1[0] + patch_size - 1, pt1[1] + patch_size - 1)
        if i == 0:
            count = count + 1
            final_bbox.append(bbox)
            if count == try_n_patches:
                return final_bbox
        else:
            flag = True
            for j in range(len(final_bbox)):
                if IOU(final_bbox[j], bbox) > iou_thresholds[count - 1]:
                    flag = False
            if flag:
                count = count + 1
                final_bbox.append(bbox)
                if count == try_n_patches:
                    return final_bbox
    return final_bbox


def crop_patches(inputs, cam, patch_size, stride, iou_thresholds):
    boxes = []
    patches = []
    for i in range(cam.size(0)):
        final_bbox = localize_bbox(cam[i], patch_size, stride, iou_thresholds)
        boxes.append(final_bbox)
        for bbox in final_bbox:
            patches.append(inputs[i, :, bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1])
    patches = torch.stack(patches, dim=0)
    return patches, boxes


def pos_lists():
    pos_list_95 = []
    size_95 = 17

    for i in range(size_95):
        for j in range(size_95):
            pos_list_95.append((i, j))

    return pos_list_95


def load_models():
    """Load the models needed for inference with their checkpoints"""
    ARCH = "densenet201"
    anchornet = AnchorNet(num_classes=1000).cuda()
    anchornet_checkpoint = torch.load("pretrained/anchornet.pth", map_location=torch.device("cpu"))
    anchornet.load_state_dict(anchornet_checkpoint)
    anchornet.eval()
    print("load AnchorNet successfully!")

    down_net = densenet201(num_classes=1000).cuda()
    checkpoint = torch.load("pretrained/densenet201_patch_95_best.pth.tar", map_location=torch.device("cpu"))["net"]
    down_net.load_state_dict(checkpoint)
    down_net.eval()
    print("load " + ARCH + " successfully!")

    resized_down_net = densenet201(num_classes=1000).cuda()
    checkpoint = torch.load("pretrained/densenet201_resized_95_best.pth.tar",map_location=torch.device("cpu"))["net"]
    resized_down_net.load_state_dict(checkpoint)
    resized_down_net.eval()
    print("load " + ARCH + " successfully!")

    return anchornet, down_net, resized_down_net


def get_imagenet_classnames():
    """Download a text file with ImageNet class names, clean and return as a map of id - name"""
    imagenet_class_names_file = "imagenet1000_clsidx_to_labels.txt"
    if not os.path.exists(imagenet_class_names_file):
        import urllib.request
        import zipfile

        urllib.request.urlretrieve(
            "https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a/archive/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5.zip",
            "tmp.zip",
        )
        zipfile.ZipFile("tmp.zip", "r").extractall(".")
        os.rename(
            f"942d3a0ac09ec9e5eb3a-238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/{imagenet_class_names_file}",
            imagenet_class_names_file,
        )
        os.remove("tmp.zip")
        os.rmdir("942d3a0ac09ec9e5eb3a-238f720ff059c1f82f368259d1ca4ffa5dd8f9f5")
    class_names = eval(open("imagenet1000_clsidx_to_labels.txt").read())
    class_names = {k: v.split(",")[0] for k, v in class_names.items()}
    return class_names

def visualize_predictions(raw_image, boxes, topk, results_dict, output_path):
    """Create a debug image showing the classified patches and the aggregated classification"""
    COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'magenta', 'cyan']

    plt.imshow(raw_image)
    for i in range(topk):
        y0, x0, y1, x1 = boxes[0][i]
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=3, edgecolor=COLORS[i], facecolor='none',
                         label=f"Patch #{1 + i}: " + results_dict[f"Patch #{i}"])
        plt.gca().add_patch(rect)
        plt.gca().axis("off")

    rect = Rectangle((0, 0), 224, 224, linewidth=3, edgecolor='k', facecolor='none',
                     label=f"Resized-input: " + results_dict[f"Resized_input"])
    plt.gca().add_patch(rect)

    plt.plot([], [], label="Agregated: " + results_dict["aggregated"], color='white')

    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    handles = handles[1:] + [handles[0]]
    labels = labels[1:] + [labels[0]]

    plt.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), fancybox=True, shadow=True, ncol=1)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()

def visualize_CAM(cam, class_name, output_path):
    plt.imshow(cam, aspect='auto')
    plt.title(f"CAM for class: {class_name}")
    plt.colorbar()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()

def plot_topk_classes(names, scores, path):
    n = len(names)
    plt.figure(figsize=[4, 4])
    plt.barh(range(n), scores, align="center")
    plt.yticks(range(n), names)
    plt.tight_layout()

    plt.savefig(path)
    plt.clf()