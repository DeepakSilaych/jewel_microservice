import replicate
import requests
from PIL import Image, ImageOps
import io
import time
import os
from openai import OpenAI
from config import REPLICATE_API_TOKEN, OPENAI_API_KEY
from image_processor import save_numbered_image

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN
client = OpenAI(api_key=OPENAI_API_KEY)

JEWELRY_FOCUSED_PROMPTS = {
    "necklaces": {
        "base_prompt": "Professional female model portrait, bare shoulders, elegant neck visible, clean white background, no jewelry, soft lighting, fashion photography",
        "composition": "chest up portrait focusing on neck area"
    },
    "earrings": {
        "base_prompt": "Professional female model headshot, hair pulled back, ears visible, profile view, clean background, no earrings, beauty lighting",
        "composition": "headshot emphasizing ears and face"
    },
    "bracelets": {
        "base_prompt": "Professional female model hands and wrists, elegant pose, clean background, no bracelets, manicured hands, soft lighting",
        "composition": "hands and wrists focus"
    },
    "rings": {
        "base_prompt": "Professional female model hands close-up, fingers extended, elegant pose, clean background, no rings, perfect manicure, studio lighting",
        "composition": "close-up hands and fingers"
    }
}

def generate_model_image(jewelry_type, size="1024x1024", jewelry_subtype=None):
    jewelry_type = jewelry_type.lower()
    print(f"Generating model image for: {jewelry_type} ({jewelry_subtype})")
    
    if jewelry_type not in JEWELRY_FOCUSED_PROMPTS:
        jewelry_type = "necklaces"
    
    prompt_data = JEWELRY_FOCUSED_PROMPTS[jewelry_type]
    base_prompt = prompt_data['base_prompt']
    
    if jewelry_subtype:
        enhanced_base_prompt = get_jewelry_specific_prompt(jewelry_type, jewelry_subtype)
    else:
        enhanced_base_prompt = base_prompt
    
    final_prompt = f"{enhanced_base_prompt}, {prompt_data['composition']}, high quality, realistic, professional photography"
    print(f"Using prompt: {final_prompt}")
    
    try:
        output = replicate.run(
            "black-forest-labs/flux-dev",
            input={
                "prompt": final_prompt,
                "go_fast": True,
                "guidance": 3.5,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 80,
                "prompt_strength": 0.8,
                "num_inference_steps": 28
            }
        )
        print("Replicate generation successful")
        
        if output and len(output) > 0:
            image_url = output[0]
            print(f"Generated image URL: {image_url}")
            
            image_response = requests.get(image_url)
            image = Image.open(io.BytesIO(image_response.content))
            
            # Save base model image with sequential numbering
            base_path = save_numbered_image(image, f"model_{jewelry_type}_base")
            print(f"Model image saved: {base_path}")
            
            # Zoom in on the center of the image
            zoomed_image = zoom_in_center(image, zoom_factor=1.2)
            
            # Save the zoomed image with sequential numbering
            zoomed_path = save_numbered_image(zoomed_image, f"model_{jewelry_type}_zoomed")
            print(f"Zoomed model image saved: {zoomed_path}")
            
            return zoomed_path
            
        else:
            raise Exception("No output received from Flux model")
            
    except Exception as e:
        print(f"Error generating model image: {str(e)}")
        print("Trying with simplified prompt...")
        
        try:
            simple_prompt = f"Professional female model {jewelry_type} photography"
            
            output = replicate.run(
                "black-forest-labs/flux-dev",
                input={
                    "prompt": simple_prompt,
                    "go_fast": True,
                    "guidance": 3.5,
                    "megapixels": "1",
                    "num_outputs": 1,
                    "aspect_ratio": "1:1",
                    "output_format": "webp",
                    "output_quality": 80,
                    "prompt_strength": 0.8,
                    "num_inference_steps": 28
                }
            )
            
            if output and len(output) > 0:
                image_url = output[0]
                image_response = requests.get(image_url)
                image = Image.open(io.BytesIO(image_response.content))
                
                # Save fallback base model image with sequential numbering
                base_path = save_numbered_image(image, f"model_{jewelry_type}_fallback_base")
                print(f"Fallback model image saved: {base_path}")
                
                # Zoom in on the center of the image
                zoomed_image = zoom_in_center(image, zoom_factor=1.2)
                
                # Save the zoomed fallback image with sequential numbering
                zoomed_path = save_numbered_image(zoomed_image, f"model_{jewelry_type}_fallback_zoomed")
                print(f"Zoomed fallback model image saved: {zoomed_path}")
                
                return zoomed_path
                
        except Exception as e2:
            print(f"Fallback also failed: {str(e2)}")
            raise Exception(f"Failed to generate model image: {str(e)}")

def get_jewelry_specific_prompt(jewelry_type, jewelry_subtype=None):
    jewelry_type = jewelry_type.lower()
    
    if jewelry_type not in JEWELRY_FOCUSED_PROMPTS:
        return JEWELRY_FOCUSED_PROMPTS["necklaces"]["base_prompt"]
    
    base_prompt = JEWELRY_FOCUSED_PROMPTS[jewelry_type]["base_prompt"]
    
    if jewelry_subtype:
        subtype_enhancements = {
            "chokers": "close neck shot, throat visible",
            "tennis necklaces": "elegant neck for delicate jewelry",
            "pendants": "chest area for pendant placement",
            "necklaces": "classic neck presentation",
            "stud earrings": "frontal view, ears clearly visible",
            "drop earrings": "profile view showing ear space",
            "hoop earrings": "clear ear outline, side lighting",
            "clip earrings": "ear structure visible",
            "ear cuffs": "outer ear area featured",
            "cocktail rings": "dramatic hand pose, fingers spread",
            "stackable rings": "multiple fingers visible",
            "halo rings": "ring finger featured",
            "band rings": "classic hand position",
            "motif rings": "artistic hand pose",
            "adjustable rings": "flexible finger positioning",
            "tennis bracelets": "wrist for delicate bracelet",
            "bangles and cuffs": "forearm and wrist featured",
            "chain bracelets": "classic wrist presentation",
            "adjustable bracelets": "versatile wrist positioning"
        }
        
        if jewelry_subtype.lower() in subtype_enhancements:
            base_prompt += f", {subtype_enhancements[jewelry_subtype.lower()]}"
    
    return base_prompt

def zoom_in_center(image, zoom_factor=1.2):
    """
    Zoom into the center of an image by cropping and resizing.
    
    Args:
        image (PIL.Image): The input image
        zoom_factor (float): The factor to zoom in by (e.g., 1.2 for 20% zoom)
        
    Returns:
        PIL.Image: The zoomed-in image
    """
    width, height = image.size
    new_width = width / zoom_factor
    new_height = height / zoom_factor
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    
    # Crop to the center
    cropped_image = image.crop((left, top, right, bottom))
    
    # Resize back to original dimensions
    return cropped_image.resize((width, height), Image.LANCZOS)