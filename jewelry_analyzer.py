import openai
import base64

# Detailed jewelry subtypes with professional descriptions including size characteristics
JEWELRY_SUBTYPES = {
    "necklaces": {
        "Necklaces": "Traditional chain or beaded necklaces worn around the neck, typically 16-30 inches in length, featuring a continuous design without prominent focal points. Length varies from princess (16-18\") to opera (28-34\") styles.",
        "Pendants": "Necklaces featuring a decorative hanging element (pendant) suspended from a chain, often showcasing gemstones, charms, or artistic designs. Pendant size typically ranges from 0.5-3 inches, with chain lengths of 16-24 inches.",
        "Chokers": "Short, close-fitting necklaces that sit snugly around the neck, typically 12-16 inches in length. Width can vary from delicate 2-5mm chains to bold 15-30mm collar styles.",
        "Tennis Necklaces": "Elegant necklaces featuring a continuous line of uniform diamonds or gemstones in a delicate setting, typically 16-18 inches long with stone sizes ranging from 1-5mm each."
    },
    "bracelets": {
        "Chain bracelets": "Traditional link-style bracelets made of connected metal chains, typically 6.5-8.5 inches in length and 2-10mm wide, featuring classic designs like Cuban, rope, or cable links.",
        "Adjustable bracelets": "Bracelets with flexible sizing mechanisms such as sliding clasps, elastic bands, or extendable chains, typically adjusting from 6-9 inches with varying widths from 1-15mm.",
        "Tennis bracelets": "Sophisticated bracelets featuring a continuous line of matched diamonds or gemstones, typically 6.5-7.5 inches long with stone sizes from 1-4mm and overall width of 3-6mm.",
        "Bangles and cuffs": "Rigid or semi-rigid circular bracelets with internal diameters of 2.25-2.75 inches. Bangles are typically 2-8mm wide, while cuffs range from 10-40mm wide with 1-2 inch openings."
    },
    "earrings": {
        "Stud earrings": "Simple, close-to-ear designs featuring a single gemstone, pearl, or decorative element, typically 2-12mm in diameter, sitting flush against the earlobe without dangling elements.",
        "Drop earrings": "Earrings that hang below the earlobe, featuring elements that dangle 5-50mm from the ear hook or post, with varying widths from delicate 2mm to statement 20mm designs.",
        "Hoop earrings": "Circular or oval-shaped earrings forming complete or partial loops, ranging from small huggie styles (8-15mm diameter) to large statement hoops (40-80mm diameter) with wire thickness of 1-4mm.",
        "Clip earrings": "Non-pierced earrings that attach using spring-loaded clips, typically 8-25mm in size with clip mechanisms adding 3-8mm to the overall depth.",
        "Ear cuffs": "Decorative pieces that wrap around the outer ear edge, typically 10-30mm in length and 2-15mm wide, designed to follow the ear's natural curve without piercings."
    },
    "rings": {
        "Halo rings": "Rings featuring a central gemstone (3-10mm) surrounded by a circle of smaller diamonds or gemstones (1-2mm each), with total face diameter typically 8-15mm and band width of 1.5-3mm.",
        "Band rings": "Simple, continuous circular rings without prominent stones, typically 1-8mm wide, including wedding bands (2-4mm), eternity bands (2-3mm), and statement bands (5-8mm).",
        "Cocktail rings": "Bold, statement rings featuring large gemstones (8-20mm) or elaborate designs, typically with face diameters of 12-25mm and substantial band widths of 3-8mm for special occasions.",
        "Motif rings": "Rings featuring specific decorative themes, symbols, or artistic patterns, with motif sizes ranging from delicate 5-8mm to bold 15-20mm designs and band widths of 1.5-5mm.",
        "Adjustable rings": "Rings with flexible sizing mechanisms, typically featuring open bands with 2-8mm gaps, allowing adjustment across 2-4 ring sizes with band widths of 1-6mm.",
        "Stackable rings": "Thin, complementary rings designed for multiple wear, typically 1-3mm wide with delicate proportions, featuring minimal stone settings under 5mm to maintain comfortable stacking."
    }

}


def analyze_jewelry_subtype(jewelry_image_path, jewelry_type, size=None):
    jewelry_type = jewelry_type.lower()
    
    if jewelry_type not in JEWELRY_SUBTYPES:
        return jewelry_type
    
    with open(jewelry_image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    subtypes_dict = JEWELRY_SUBTYPES[jewelry_type]
    
    subtype_descriptions = ""
    for subtype, description in subtypes_dict.items():
        subtype_descriptions += f"• {subtype}: {description}\n"
    
    size_context = ""
    size_hints = ""
    if size:
        size_info = parse_size_information(size)
        size_hints = get_size_classification_hints(jewelry_type, size_info)
        
        size_context = f"\nJEWELRY SIZE INFORMATION: {size}"
        if size_hints:
            size_context += f"\nSIZE-BASED CLASSIFICATION HINTS: {size_hints}"
        size_context += f"\nConsider this size information when determining the most appropriate subtype. Use the size specifications in the descriptions above to guide your classification.\n"
    
    classification_prompt = f"""You are a certified gemologist and jewelry expert with over 20 years of experience in fine jewelry identification and classification. Your expertise includes detailed knowledge of jewelry construction, design elements, size specifications, and industry-standard terminology.

TASK: Analyze the provided {jewelry_type} image and classify it into the most appropriate subtype based on visual characteristics and size information.

JEWELRY CATEGORY: {jewelry_type.title()}
{size_context}
AVAILABLE SUBTYPES AND DESCRIPTIONS:
{subtype_descriptions}

ANALYSIS GUIDELINES:
1. Examine the jewelry's structural design and key identifying features
2. Consider the mounting style, setting type, and overall construction
3. Pay special attention to size characteristics and proportions mentioned in the descriptions
4. Focus on distinguishing size-based features that differentiate between subtypes
5. Cross-reference visual elements with the provided size information
6. Consider scale indicators in the image (if any) to estimate dimensions

RESPONSE FORMAT:
- Return ONLY the exact subtype name from the list above
- Do not include explanations, descriptions, or additional text
- Ensure the response matches one of the provided subtypes exactly

Please analyze the image and size information to provide your expert classification."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional jewelry expert and certified gemologist specializing in precise jewelry classification. You provide accurate, concise responses based on visual analysis of jewelry pieces."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": classification_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=100,
            temperature=0.1  # Low temperature for consistent, accurate classification
        )
        
        result = response.choices[0].message.content.strip()
        
        # Validate that the result is one of the expected subtypes
        if result in subtypes_dict:
            return result
        else:
            # Fallback: find the closest match
            for subtype in subtypes_dict.keys():
                if subtype.lower() in result.lower() or result.lower() in subtype.lower():
                    return subtype
            
            # If no match found, return the first subtype as fallback
            return list(subtypes_dict.keys())[0]
            
    except Exception as e:
        print(f"Error in jewelry analysis: {str(e)}")
        return jewelry_type


def get_jewelry_subtype_description(jewelry_type, subtype):
    """
    Get detailed description for a specific jewelry subtype.
    
    Args:
        jewelry_type (str): Type of jewelry (necklaces, bracelets, earrings, rings)
        subtype (str): Specific subtype
    
    Returns:
        str: Detailed description of the subtype
    """
    jewelry_type = jewelry_type.lower()
    
    if jewelry_type in JEWELRY_SUBTYPES and subtype in JEWELRY_SUBTYPES[jewelry_type]:
        return JEWELRY_SUBTYPES[jewelry_type][subtype]
    
    return "Description not available for this jewelry subtype."


def get_all_subtypes_for_category(jewelry_type):
    """
    Get all available subtypes for a specific jewelry category.
    
    Args:
        jewelry_type (str): Type of jewelry (necklaces, bracelets, earrings, rings)
    
    Returns:
        dict: Dictionary of subtypes and their descriptions
    """
    jewelry_type = jewelry_type.lower()
    
    if jewelry_type in JEWELRY_SUBTYPES:
        return JEWELRY_SUBTYPES[jewelry_type]
    
    return {}


def parse_size_information(size_str):
    """
    Parse and normalize size information for better analysis.
    
    Args:
        size_str (str): Size string (e.g., "2.6 x 2.8 cm", "15mm diameter", "7 inches")
    
    Returns:
        dict: Normalized size information with dimensions and units
    """
    if not size_str:
        return {}
    
    size_info = {"original": size_str.strip()}
    size_lower = size_str.lower().strip()
    
    # Extract dimensions and units
    import re
    
    # Pattern for dimensions like "2.6 x 2.8 cm" or "15mm x 20mm"
    dimension_pattern = r'(\d+\.?\d*)\s*[x×]\s*(\d+\.?\d*)\s*(mm|cm|inch|inches|in)?'
    dimension_match = re.search(dimension_pattern, size_lower)
    
    if dimension_match:
        size_info["width"] = float(dimension_match.group(1))
        size_info["height"] = float(dimension_match.group(2))
        size_info["unit"] = dimension_match.group(3) or "mm"
    else:
        # Pattern for single dimension like "15mm diameter" or "7 inches"
        single_pattern = r'(\d+\.?\d*)\s*(mm|cm|inch|inches|in)'
        single_match = re.search(single_pattern, size_lower)
        
        if single_match:
            size_info["dimension"] = float(single_match.group(1))
            size_info["unit"] = single_match.group(2)
            
            # Determine if it's diameter, length, width based on context
            if "diameter" in size_lower or "diam" in size_lower:
                size_info["type"] = "diameter"
            elif "length" in size_lower or "long" in size_lower:
                size_info["type"] = "length"
            elif "width" in size_lower or "wide" in size_lower:
                size_info["type"] = "width"
    
    return size_info


def get_size_classification_hints(jewelry_type, size_info):
    """
    Generate size-based classification hints for specific jewelry types.
    
    Args:
        jewelry_type (str): Type of jewelry
        size_info (dict): Parsed size information
    
    Returns:
        str: Size-based classification hints
    """
    if not size_info:
        return ""
    
    hints = []
    jewelry_type = jewelry_type.lower()
    
    if jewelry_type == "earrings":
        if "width" in size_info and "height" in size_info:
            width, height = size_info["width"], size_info["height"]
            if width == height and width <= 12:  # Small and round
                hints.append("Size suggests stud earrings (typically ≤12mm)")
            elif height > width * 1.5:  # Tall and narrow
                hints.append("Proportions suggest drop earrings (height > width)")
            elif abs(width - height) < 3 and width > 15:  # Large and round
                hints.append("Large circular dimensions suggest hoop earrings (>15mm)")
        elif "dimension" in size_info:
            dim = size_info["dimension"]
            if size_info.get("type") == "diameter" and dim > 15:
                hints.append("Large diameter suggests hoop earrings")
            elif dim <= 12:
                hints.append("Small size suggests stud earrings")
    
    elif jewelry_type == "rings":
        if "width" in size_info:
            width = size_info["width"]
            if width <= 3:
                hints.append("Narrow width suggests stackable or band rings")
            elif width >= 5:
                hints.append("Wide band suggests cocktail or statement rings")
        
        if "height" in size_info and "width" in size_info:
            face_size = max(size_info["height"], size_info["width"])
            if face_size >= 12:
                hints.append("Large face size suggests cocktail rings")
            elif face_size <= 8:
                hints.append("Small face size suggests delicate or stackable rings")
    
    elif jewelry_type == "necklaces":
        if "dimension" in size_info and size_info.get("type") == "length":
            length = size_info["dimension"]
            unit = size_info.get("unit", "")
            
            # Convert to inches for comparison
            if "cm" in unit:
                length_inches = length / 2.54
            elif "mm" in unit:
                length_inches = length / 25.4
            else:
                length_inches = length
                
            if length_inches <= 16:
                hints.append("Short length suggests chokers")
            elif length_inches >= 28:
                hints.append("Long length suggests opera-style necklaces")
    
    return " ".join(hints) if hints else ""