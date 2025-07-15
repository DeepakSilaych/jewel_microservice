import openai
import os
import base64

def generate_kontext_prompt(jewelry_type, jewelry_subtype, model_image_path, jewelry_size, placement_position=None):
    """
    Generate a precise Kontext prompt for jewelry placement following best practices:
    - Be specific with exact descriptions
    - Preserve intentionally what should remain unchanged  
    - Use clear action verbs and direct subject naming
    - Control composition explicitly
    """
    
    # Create specific prompt based on jewelry type
    jewelry_context = {
        "earrings": {
            "placement": "positioned naturally at the ear lobe",
            "lighting": "with soft ambient lighting that creates subtle reflections on the metal surface",
            "integration": "seamlessly integrated as if naturally worn"
        },
        "necklaces": {
            "placement": "positioned around the neck following the natural neckline", 
            "lighting": "with gentle lighting that highlights the chain or pendant details",
            "integration": "naturally draped and positioned as if carefully placed"
        },
        "bracelets": {
            "placement": "positioned around the wrist with natural draping",
            "lighting": "with lighting that enhances the bracelet's texture and materials", 
            "integration": "fitted naturally as if worn in daily life"
        },
        "rings": {
            "placement": "positioned on the appropriate finger with realistic sizing",
            "lighting": "with focused lighting that showcases the ring's details and gemstones",
            "integration": "naturally fitted as if custom-sized for the hand"
        }
    }
    
    context = jewelry_context.get(jewelry_type, jewelry_context["necklaces"])
    
    # Use O3 model with image context for advanced prompt generation
    try:
        # Prepare message content with image if available
        message_content = [
            {
                "type": "text",
                "text": f"""Create a precise Kontext image editing prompt for this model image:

JEWELRY DETAILS:
- Type: {jewelry_type}
- Subtype: {jewelry_subtype} 
- Size: {jewelry_size}
- Placement: {context['placement']}

KONTEXT REQUIREMENTS:
1. Be specific: Use exact colors, detailed descriptions
2. Preserve: "while maintaining the same facial features, pose, lighting, background"
3. Name directly: "the woman" not pronouns
4. Action verb: "Add" not "transform"
5. Max 100 words

Structure: "Add [jewelry description] {context['placement']} while maintaining [preserve elements]. {context['integration']} {context['lighting']}."""
            }
        ]
        
        # Add image if model image path exists
        if model_image_path and os.path.exists(model_image_path):
            import base64
            with open(model_image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            message_content.insert(0, {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        response = openai.chat.completions.create(
            model="gpt-4o",  # Use GPT-4o for vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": message_content
                }
            ],
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error generating Kontext prompt: {e}")
        # Fallback to manual prompt construction
        return create_fallback_kontext_prompt(jewelry_type, jewelry_subtype, jewelry_size, context)

def create_fallback_kontext_prompt(jewelry_type, jewelry_subtype, jewelry_size, context):
    """Fallback method to create Kontext prompt when API fails"""
    
    # Sample prompts based on jewelry type
    sample_prompts = {
        "earrings": f"Add elegant {jewelry_subtype} earrings in polished silver metal {context['placement']}, while maintaining the same facial features, hair style, pose, and background lighting. The earrings should appear {context['integration']} {context['lighting']}.",
        
        "necklaces": f"Add a delicate {jewelry_subtype} necklace in refined metal finish {context['placement']}, while preserving the exact facial expression, clothing, and camera angle. The necklace should be {context['integration']} {context['lighting']}.",
        
        "bracelets": f"Add a sophisticated {jewelry_subtype} bracelet in premium metal finish {context['placement']}, while keeping the same hand position, clothing, and background unchanged. The bracelet should appear {context['integration']} {context['lighting']}.",
        
        "rings": f"Add an elegant {jewelry_subtype} ring in polished metal finish {context['placement']}, while maintaining the exact hand pose, skin tone, and surrounding elements. The ring should be {context['integration']} {context['lighting']}."
    }
    
    return sample_prompts.get(jewelry_type, sample_prompts["necklaces"])

def generate_context_prompt(jewelry_type, base_prompt, size):
    """Legacy function for backward compatibility"""
    return generate_kontext_prompt(jewelry_type, jewelry_type, "", size)

# Sample Kontext prompts for reference
SAMPLE_KONTEXT_PROMPTS = {
    "earrings": """Add elegant emerald and white diamond cluster drop earrings in polished white gold positioned naturally at the woman's right earlobe, while maintaining the same facial features, hair style, pose, and background lighting. The earrings should appear seamlessly integrated as if naturally worn with soft ambient lighting that creates subtle reflections on the metal surface.""",
    
    "necklaces": """Add a delicate pearl and diamond pendant necklace in refined platinum finish positioned around the neck following the natural neckline, while preserving the exact facial expression, clothing, and camera angle. The necklace should be naturally draped and positioned as if carefully placed with gentle lighting that highlights the pendant details.""",
    
    "bracelets": """Add a sophisticated tennis bracelet with alternating diamonds in premium white gold finish positioned around the wrist with natural draping, while keeping the same hand position, clothing, and background unchanged. The bracelet should appear fitted naturally as if worn in daily life with lighting that enhances the bracelet's sparkle.""",
    
    "rings": """Add an elegant solitaire diamond engagement ring in polished platinum finish positioned on the ring finger with realistic sizing, while maintaining the exact hand pose, skin tone, and surrounding elements. The ring should be naturally fitted as if custom-sized for the hand with focused lighting that showcases the diamond's brilliance."""
} 