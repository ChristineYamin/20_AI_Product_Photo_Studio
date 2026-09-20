📸 AI Product Photo Studio

AI Product Photo Studio is a Streamlit application that transforms ordinary product photos into polished promotional images. Users select a product with guided clicks, remove its original background using SAM 2, generate a new AI background, and fine-tune the final composition through interactive photo controls.

🔗 Live app: project20-photo-studio.streamlit.app

Overview

Creating clean product photography normally requires a studio, careful lighting, and manual editing. This project provides a simpler workflow inside one web application:

Upload a product image.

Click at least two areas inside the product.

Let SAM 2 isolate the selected object.

Choose the product type, camera view, and background style.

Generate an empty AI background with Cloudflare Workers AI.

Adjust the product and download the completed promotional image.

Features

Guided product selection with visible, numbered click points

SAM 2 segmentation for interactive background removal

AI background generation through Cloudflare Workers AI and FLUX.1 Schnell

Upright and top-down compositions for products and food photography

Product-aware background prompts for perfume, cosmetics, jewellery, food, beverages, electronics, toys, fashion accessories, and other products

Multiple visual styles, including Luxury, Studio, Cinematic, Showroom, Nature, Café, Playful, Rustic, and Futuristic

Generated background history with up to three recent alternatives

Solid-colour background option

Interactive photo controls for size, position, brightness, contrast, saturation, sharpness, shadow strength, and shadow softness

Multiple export formats for Instagram, TikTok, stories, and websites

Before-and-final comparison with equally sized previews

PNG download of the finished product image

Responsive Streamlit interface with a custom playful visual theme

Technology Stack

Component

Technology

User interface

Streamlit

Product segmentation

Meta SAM 2 via Hugging Face Transformers

Background generation

Cloudflare Workers AI — FLUX.1 Schnell

Image processing

Pillow

Deep-learning runtime

PyTorch

Interactive selection

streamlit-image-coordinates

HTTP communication

Requests

Application Workflow

flowchart TD
    A[Upload product photo] --> B[Click inside the product]
    B --> C[SAM 2 creates product mask]
    C --> D[Choose product type and style]
    D --> E[Generate AI background]
    E --> F[Position and enhance product]
    F --> G[Compare and download result]

Installation

1. Clone the repository

git clone https://github.com/ChristineYamin/20_AI_Product_Photo_Studio.git
cd 20_AI_Product_Photo_Studio

If your repository uses a different URL or folder name, replace the values above accordingly.

2. Create a virtual environment

On Windows:

python -m venv venv
venv\Scripts\activate

On macOS or Linux:

python -m venv venv
source venv/bin/activate

3. Install the dependencies

pip install -r requirements.txt

4. Configure environment variables

Create a .env file in the project directory:

HF_TOKEN=your_hugging_face_token
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id

The Hugging Face token is used when downloading the SAM 2 model. The Cloudflare credentials are required for AI background generation.

Never commit .env or .streamlit/secrets.toml to GitHub.

5. Run the application

streamlit run app.py

Then open http://localhost:8501 in your browser.

Streamlit Community Cloud Deployment

Add the following values to App settings → Secrets in Streamlit Community Cloud:

HF_TOKEN = "your_hugging_face_token"
CLOUDFLARE_API_TOKEN = "your_cloudflare_api_token"
CLOUDFLARE_ACCOUNT_ID = "your_cloudflare_account_id"

Keep all tokens private. The deployed application reads them as environment variables and does not display them to users.

Usage Tips

Click clearly inside the product rather than on the surrounding background.

Use points across different product areas when the object contains separate shapes or complex edges.

Choose Top-down for flat-lay food and tabletop products.

Choose Upright for bottles, electronics, toys, cosmetics, and standing products.

Adjust the vertical position and shadow controls so the product appears to rest naturally on the generated surface.

If an AI background includes an unwanted object, generate another variation and select it from the background history.

Output Formats

Format

Resolution

Square — Instagram

1080 × 1080

Portrait — Instagram

1080 × 1350

Story — Instagram/TikTok

1080 × 1920

Landscape — Website

1200 × 800

Limitations

Segmentation quality depends on the selected points, image clarity, and contrast between the product and its original background.

Transparent, reflective, furry, or unusually shaped products may require additional selection points.

AI-generated backgrounds can occasionally contain unwanted objects or imperfect surfaces.

Initial startup may take longer while the SAM 2 model is downloaded and loaded.

Background generation depends on Cloudflare Workers AI availability and account usage limits.

Future Improvements

Negative selection points for excluding unwanted regions

More precise mask refinement controls

Automatic lighting and colour matching between the product and background

Additional background styles and product-specific presets

Faster model loading and inference optimization

Author

Created by Yamin as Project 20 of the 23 Projects at 23 portfolio challenge.

GitHub: ChristineYamin

Live application: AI Product Photo Studio