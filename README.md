# Jewelry Model Image Generator

A Python agent that creates model images with jewelry using AI.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```bash
export OPENAI_API_KEY="your_openai_key"
export REPLICATE_API_TOKEN="your_replicate_token"
```

## Usage

```python
from main import create_jewelry_model_image

result = create_jewelry_model_image(
    jewelry_image_path="ring.jpg",
    jewelry_type="rings",
    prompt="elegant fashion model",
    size="1024x1024"
)
```

## Supported Jewelry Types

- **necklaces**: Necklaces, Pendants, Chokers, Tennis Necklaces
- **bracelets**: Chain bracelets, Adjustable bracelets, Tennis bracelets, Bangles and cuffs
- **earrings**: Stud earrings, Drop earrings, Hoop earrings, Clip earrings, Ear cuffs
- **rings**: Halo rings, Band rings, Cocktail rings, Motif rings, Adjustable rings, Stackable rings

## Files

- `main.py` - Main orchestrator
- `jewelry_analyzer.py` - Analyzes jewelry subtype
- `model_generator.py` - Creates base model image
- `prompt_generator.py` - Generates context prompts
- `image_merger.py` - Merges images with Flux
- `config.py` - API configuration
# jewel_microservice
