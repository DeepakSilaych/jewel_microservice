import os
from PIL import Image
from config import setup_apis
from jewelry_analyzer import analyze_jewelry_subtype
from model_generator import generate_model_image
from prompt_generator import generate_kontext_prompt
from image_merger import merge_images_with_flux_kontext
from image_processor import resize_and_overlay

def create_jewelry_model_image(jewelry_image_path, jewelry_type, prompt, size):
    """Main function to create jewelry model images with complete image saving pipeline"""
    setup_apis()
    
    print("\n🎯 STARTING COMPLETE JEWELRY PIPELINE")
    print("=" * 60)
    
    # Step 1: Analyze jewelry
    jewelry_subtype = analyze_jewelry_subtype(jewelry_image_path, jewelry_type, size)
    print(f"1️⃣ Identified jewelry subtype: {jewelry_subtype}")
    
    # Step 2: Generate model image (saves: 1.model, 2.model_zoomed) 
    model_image = generate_model_image(jewelry_type, "1024x1024", jewelry_subtype)
    print(f"2️⃣ Generated model image: {model_image}")
    
    # Step 3: Crop jewelry (saves: 3.jewelry_cropped)
    from image_processor import crop_jewelry_image, save_numbered_image
    cropped_jewelry = crop_jewelry_image(jewelry_image_path)
    print(f"3️⃣ Cropped jewelry saved: {cropped_jewelry}")
    
    # Step 4: Traditional overlay to create composite (saves: 4.model_with_overlayed_jewelry)
    from image_processor import resize_and_overlay
    overlay_result = resize_and_overlay(model_image, jewelry_image_path, size, jewelry_type)
    print(f"4️⃣ Composite image created: {overlay_result}")
    
    # Step 5: Generate Kontext prompt with composite image context
    kontext_prompt = generate_kontext_prompt(jewelry_type, jewelry_subtype, overlay_result, size)
    print(f"5️⃣ Generated Kontext prompt: {kontext_prompt}")
    
    # Step 6: Use Flux Kontext Pro with composite image (saves: 5.model_with_site)
    kontext_result = merge_images_with_flux_kontext(overlay_result, kontext_prompt, f"{jewelry_type}_with_site")
    print(f"6️⃣ Kontext result: {kontext_result}")
    
    # Step 7: Choose final result (saves: 6.final)
    if kontext_result:
        # Use Kontext result as final (enhanced composite)
        final_img = Image.open(kontext_result)
        final_path = save_numbered_image(final_img, f"final_{jewelry_type}_kontext_enhanced")
        print(f"7️⃣ Final image (Kontext Enhanced): {final_path}")
        final_image = final_path
    else:
        # Use original overlay result as final
        final_img = Image.open(overlay_result)  
        final_path = save_numbered_image(final_img, f"final_{jewelry_type}_overlay")
        print(f"7️⃣ Final image (Original Overlay): {final_path}")
        final_image = final_path
    
    print("\n✅ COMPLETE PIPELINE FINISHED")
    print("📁 All 6 images saved in img/ directory")
    
    return {
        "subtype": jewelry_subtype, 
        "model_image": model_image,
        "cropped_jewelry": cropped_jewelry,
        "kontext_result": kontext_result,
        "overlay_result": overlay_result, 
        "final_image": final_image,
        "kontext_prompt": kontext_prompt
    }

def main(jewelry_image_path, size="medium"):
    """Simplified main function for testing"""
    
    # Determine jewelry type from filename or analyze
    if "earring" in jewelry_image_path.lower():
        jewelry_type = "earrings"
    elif "necklace" in jewelry_image_path.lower():
        jewelry_type = "necklaces"
    elif "bracelet" in jewelry_image_path.lower():
        jewelry_type = "bracelets"
    elif "ring" in jewelry_image_path.lower():
        jewelry_type = "rings"
    else:
        jewelry_type = "earrings"  # Default
    
    prompt = "elegant fashion model"
    return create_jewelry_model_image(jewelry_image_path, jewelry_type, prompt, size)

if __name__ == "__main__":
    jewelry_image = "ring.png" 
    jewelry_type = "rings"
    prompt = "create a indian model wearing this ring"
    size = "1cm"
    
    result = create_jewelry_model_image(jewelry_image, jewelry_type, prompt, size)
    print(f"\n🏁 COMPLETE PIPELINE RESULTS:")
    print(f"📊 Jewelry Analysis:")
    print(f"   - Subtype: {result['subtype']}")
    print(f"   - Kontext Prompt: {result['kontext_prompt'][:100]}...")
    print(f"\n📁 6 Saved Images:")
    print(f"   1. Model Base: (auto-saved in model generation)")
    print(f"   2. Model Zoomed: {result['model_image']}")
    print(f"   3. Jewelry Cropped: {result['cropped_jewelry']}")
    print(f"   4. Composite (Model + Jewelry): {result['overlay_result']}")
    print(f"   5. Kontext Enhanced: {result['kontext_result']}")
    print(f"   6. Final: {result['final_image']}")