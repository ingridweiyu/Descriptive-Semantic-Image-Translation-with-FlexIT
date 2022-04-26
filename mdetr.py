import torch
from PIL import Image
import requests
import torchvision.transforms as T
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
from skimage.measure import find_contours

from matplotlib import patches, lines
from matplotlib.patches import Polygon

# torch.set_grad_enabled(False);

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def box_cxcywh_to_xyxy_margin_tens(x, margin):
    x_c, y_c, w, h = x.unbind(1)
    bb_tens = [(x_c - (0.5 * w) - margin), (y_c - (0.5 * h) - margin),
         (x_c + (0.5 * w) + margin), (y_c + (0.5 * h) + margin)]
    return torch.stack(bb_tens, dim=1)

def box_cxcywh_to_xyxy_margin_coor(x, margin):
    x_c, y_c, w, h = x.unbind(1)
    bb_coor = [(x_c - (0.5 * w) - margin).item(), (y_c - (0.5 * h) - margin).item(),
         (x_c + (0.5 * w) + margin).item(), (y_c + (0.5 * h) + margin).item()]
    return bb_coor

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy_margin_tens(out_bbox, 0.02)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def plot_results(pil_img, scores, boxes, labels, masks=None):
    plt.figure(figsize=(16,10))
    np_image = np.array(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if masks is None:
      masks = [None for _ in range(len(scores))]
    assert len(scores) == len(boxes) == len(labels) == len(masks)
    for s, (xmin, ymin, xmax, ymax), l, mask, c in zip(scores, boxes.tolist(), labels, masks, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{l}: {s:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

        if mask is None:
          continue
        np_image = apply_mask(np_image, mask, c)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
          # Subtract the padding and flip (y, x) to (x, y)
          verts = np.fliplr(verts) - 1
          p = Polygon(verts, facecolor="none", edgecolor=c)
          ax.add_patch(p)


    plt.imshow(np_image)
    plt.axis('off')
    plt.show()


def add_res(results, ax, color='green'):
    #for tt in results.values():
    if True:
        bboxes = results['boxes']
        labels = results['labels']
        scores = results['scores']
        #keep = scores >= 0.0
        #bboxes = bboxes[keep].tolist()
        #labels = labels[keep].tolist()
        #scores = scores[keep].tolist()
    #print(torchvision.ops.box_iou(tt['boxes'].cpu().detach(), torch.as_tensor([[xmin, ymin, xmax, ymax]])))
    
    colors = ['purple', 'yellow', 'red', 'green', 'orange', 'pink']
    
    for i, (b, ll, ss) in enumerate(zip(bboxes, labels, scores)):
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, color=colors[i], linewidth=3))
        cls_name = ll if isinstance(ll,str) else CLASSES[ll]
        text = f'{cls_name}: {ss:.2f}'
        print(text)
        ax.text(b[0], b[1], text, fontsize=15, bbox=dict(facecolor='white', alpha=0.8))

model_pc = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_phrasecut', pretrained=True, return_postprocessor=False)
# model_pc = model_pc.cuda()
model_pc.eval()

def return_bounding_box_max(im, caption):
  # mean-std normalize the input image (batch-size: 1)
  # img = transform(im).unsqueeze(0).cuda()
  img = transform(im).unsqueeze(0)

  # propagate through the model
  outputs = model_pc(img, [caption])

  # keep only max confidence prediction(s)
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  #keep = (probas > 0.9).cpu()
  index = torch.argmax(probas)
  keep_np = np.full(100, False)
  keep_np[index] = True
  keep = torch.from_numpy(keep_np)

  # convert boxes from [0; 1] to image scales
  bboxes_xyxy = box_cxcywh_to_xyxy_margin_coor(outputs['pred_boxes'].cpu()[0, keep], 0.02)

  return bboxes_xyxy


def plot_inference_segmentation(im, caption):
  # mean-std normalize the input image (batch-size: 1)
  # img = transform(im).unsqueeze(0).cuda()
  img = transform(im).unsqueeze(0)

  # propagate through the model
  outputs = model_pc(img, [caption])

  # keep only prediction(s) with 0.9+ 
  probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
  #keep = (probas > 0.9).cpu()
  index = torch.argmax(probas)
  keep_np = np.full(100, False)
  keep_np[index] = True
  keep = torch.from_numpy(keep_np)

  # convert boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)

  # Interpolate masks to the correct size
  w, h = im.size
  masks = F.interpolate(outputs["pred_masks"], size=(h, w), mode="bilinear", align_corners=False)
  masks = masks.cpu()[0, keep].sigmoid() > 0.5

  tokenized = model_pc.detr.transformer.tokenizer.batch_encode_plus([caption], padding="longest", return_tensors="pt").to(img.device)

  # Extract the text spans predicted by each box
  positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
  predicted_spans = defaultdict(str)
  for tok in positive_tokens:
    item, pos = tok
    if pos < 255:
        span = tokenized.token_to_chars(0, pos)
        predicted_spans [item] += " " + caption[span.start:span.end]

  labels = [predicted_spans [k] for k in sorted(list(predicted_spans .keys()))]
  plot_results(im, probas[keep], bboxes_scaled, labels, masks)
  return outputs

def get_bb(image, text):
    im = Image.open(image)
    width = im.size[0]
    height = im.size[1]
    bb = return_bounding_box_max(im, text)
    x1 = int(bb[0]*width)
    y1 = int(bb[1]*height)
    x2 = int(bb[2]*width)
    y2 = int(bb[3]*height)
    
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)
    
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, width)
    ymax = min(ymax, height)
    
    crop_width = xmax - xmin
    crop_height = ymax - ymin
    square = max(crop_width, crop_height)
    
    xmax_try = xmin + square
    ymax_try = ymin + square
    if xmax_try > width or ymax_try > height:
        xmin_try = xmax - square
        ymin_try = ymax - square
        
        if xmin_try < 0 or ymin_try < 0:
            pass
        else:
            xmin = xmin_try
            ymin = ymin_try
    else:
        xmax = xmax_try
        ymax = ymax_try
    
    return [xmin, ymin, xmax, ymax]