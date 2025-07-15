import replicate
import requests
from PIL import Image
import io
import os
from image_processor import save_numbered_image

def merge_images_with_flux_kontext(model_image_path, kontext_prompt, output_name="jewelry_kontext_result"):
    """
    Use Flux Kontext Pro to add jewelry to model image based on Kontext prompt
    """
    
    # Use Flux Kontext Pro model with file path
    try:
        output = replicate.run(
            "black-forest-labs/flux-kontext-pro",
            input={
                "prompt": kontext_prompt,
                "input_image": open(model_image_path, "rb"),  # Pass file handle
                "aspect_ratio": "match_input_image",
                "output_format": "jpg",
                "safety_tolerance": 2,
                "prompt_upsampling": False
            }
        )
        
        print(f"Kontext output: {output}")
        
        # Handle different output types from Replicate
        result_url = None
        if output:
            if hasattr(output, 'url'):  # FileOutput object
                result_url = output.url
            elif isinstance(output, str):  # Direct URL
                result_url = output
            elif isinstance(output, list) and len(output) > 0:  # List of URLs
                result_url = output[0]
        
        if result_url:
            response = requests.get(result_url)
            final_image = Image.open(io.BytesIO(response.content))
            
            # Save with sequential numbering
            output_path = save_numbered_image(final_image, f"{output_name}_kontext")
            print(f"✅ Kontext result saved: {output_path}")
            return output_path
        else:
            print("❌ No valid output URL from Kontext model")
            return None
            
    except Exception as e:
        print(f"❌ Kontext error: {e}")
        return None

def merge_images_with_flux(model_image_path, jewelry_image_path, context_prompt, size):
    """Legacy function for backward compatibility"""
    with open(model_image_path, "rb") as f:
        model_image = f.read()
    
    with open(jewelry_image_path, "rb") as f:
        jewelry_image = f.read()
    
    output = replicate.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": context_prompt,
            "image": model_image,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "webp",
            "guidance_scale": 3.5,
            "num_inference_steps": 28,
            "output_quality": 80,
        }
    )
    
    result_url = output[0]
    
    response = requests.get(result_url)
    final_image = Image.open(io.BytesIO(response.content))
    
    output_path = "final_jewelry_model.webp"
    final_image.save(output_path)
    
    return output_path 