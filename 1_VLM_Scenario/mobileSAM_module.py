import sys
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import warnings
# Suppress annoying warnings.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add MobileSAM to path
sys.path.append('./mobilesam')
from mobilesam.mobile_sam.automatic_mask_generator import SamAutomaticMaskGenerator
from mobilesam.mobile_sam import sam_model_registry


model_type = "vit_t" # Change to "vit_b" or "vit_l" for larger models
model_path = "mobilesam/weights/mobile_sam.pt"
mobile_sam = sam_model_registry[model_type](checkpoint=model_path)

# Check for available devices and set the device accordingly
if torch.backends.mps.is_available(): # MACOS
    device = 'mps'
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available(): # NVIDIA GPU. Best case.
    device = 'cuda'
    print("Using CUDA")
else:
    device = 'cpu' # CPU fallback. VM usually runs on CPU. This is my case.
    print("Using CPU")

# Load MobileSAM model
mobile_sam.to(device)
mobile_sam.eval()
# Optimized parameters for different devices
if device == 'cpu':
    mask_gen = SamAutomaticMaskGenerator(
        mobile_sam,
        points_per_side=16,  # Reduced for CPU
        pred_iou_thresh=0.8,
        stability_score_thresh=0.85,
        crop_n_layers=0,  # Reduced for CPU
        min_mask_region_area=200,
    )
else:
    mask_gen = SamAutomaticMaskGenerator(
        mobile_sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        min_mask_region_area=100,
    )

# Load CLIP model
model_name = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"  # Change to "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M" for larger model
clip = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def extract_bboxes_from_masks(image):
    masks = mask_gen.generate(image)
    boxes = []
    for m in masks:
        x, y, w, h = m["bbox"]
        boxes.append(((x, y), (x + w, y + h)))
    return boxes

def crop_boxes(image, boxes):
    crops = []
    for (x1, y1), (x2, y2) in boxes:
        # Add bounds checking to prevent invalid crops
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:  # Valid crop dimensions
            crop = Image.fromarray(image[y1:y2, x1:x2])
            crops.append(crop)
        else:
            # Add a dummy crop to maintain index alignment
            crops.append(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)))
    return crops

def match_prompt(prompt, crops):
    if not crops:
        return 0
    
    text_inputs = processor(text=[prompt], return_tensors="pt", padding=True) # process text input
    image_inputs = processor(images=crops, return_tensors="pt", padding=True) # process image input
    
    with torch.no_grad():
        # Visual-language matching
        text_features = clip.get_text_features(**text_inputs)
        image_features = clip.get_image_features(**image_inputs)
        
        # Normalize features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        sims = (text_features @ image_features.T).squeeze(0)
    
    best_idx = torch.argmax(sims).item()
    return best_idx

def draw_box(image, box, prompt):
    (x1, y1), (x2, y2) = box
    out = image.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(out, prompt, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out

if __name__ == "__main__":
    import time
    start = time.time()
    prompt = "Pick the controller"
    image_path = "/home/costin/Documents/Github/ComputerVision/1_VLM_Scenario/VLM_Scenario-image.jpeg"
    
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    # image = cv2.resize(image, (1024, 1024))
    
    boxes = extract_bboxes_from_masks(image)
    crops = crop_boxes(image, boxes)
    best_idx = match_prompt(prompt, crops)
    # print(f"Best match index: {best_idx}")
    
    if best_idx < len(boxes):
        result = draw_box(image, boxes[best_idx], prompt)
        cv2.imwrite("result_image.jpg", result)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Error: Index {best_idx} is out of range for {len(boxes)} boxes")
    end = time.time()
    print(f"Execution time: {end - start:.2f} seconds")