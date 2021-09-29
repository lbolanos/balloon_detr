import json
import base64
import io
from PIL import Image
import yaml
import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection
from transformers import DetrFeatureExtractor


class Detr(pl.LightningModule):

    def __init__(self, id2label):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                            num_labels=len(id2label),
                                                            ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs


def init_context(context):
    context.logger.info("Init context...  0%")

    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    context.user_data.labels = labels

    model_path = "/tmp/tfmodel/ballon_model.pth"
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")

    context.user_data.feature_extractor = feature_extractor

    model_handler = Detr(labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_handler.load_state_dict(torch.load(model_path, device))
    context.user_data.model_handler = model_handler

    context.logger.info("Init context...100%")


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def handler(context, event):
    context.logger.info("Run detr model")
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.5))
    image = Image.open(buf)

    encoding = context.user_data.feature_extractor(image, return_tensors="pt")
    encoding.keys()

    outputs = context.user_data.model_handler(**encoding)

    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    results = []
    for p, (xmin, ymin, xmax, ymax) in zip(probas, bboxes_scaled.tolist()):
        cl = p.argmax()
        # text = f'{context.user_data.labels[cl.item()]}: {p[cl]:0.2f}'
        obj_score = p[cl]
        obj_label = context.user_data.labels.get(cl.item(), "unknown")
        results.append({
            "confidence": str(obj_score),
            "label": obj_label,
            "points": [xmin, ymin, xmax, ymax],
            "type": "rectangle",
        })

    return context.Response(body=json.dumps(results), headers={},
                            content_type='application/json', status_code=200)
