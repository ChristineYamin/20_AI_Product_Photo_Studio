# 📸 Project 20: AI Product Photo Studio

An AI-powered product photography application that transforms ordinary product photos into polished promotional images.

The application removes the original background, generates a new scene from a text prompt, and intelligently places the real product into the generated environment.

## 🎯 Project Goal

Professional product photography can require suitable backgrounds, lighting, editing skills, and expensive equipment.

This project explores how computer vision, generative AI, and image processing can simplify that workflow while preserving the appearance of the original product.

## ✨ Features

- Upload JPG, JPEG, or PNG product images
- Automatically remove image backgrounds
- Apply a custom solid-colour background
- Generate realistic backgrounds from text prompts
- Automatically detect and crop the product
- Adjust product size and position
- Adjust product brightness and contrast
- Generate a soft contact shadow
- Export images for different platforms
- Download the finished image as a PNG

## 🧠 AI and Computer Vision Pipeline

1. The user uploads a product photograph.
2. `rembg` separates the foreground product from its background.
3. The alpha mask is used to identify the product's bounding box.
4. FLUX.1-schnell generates a new background from the user's prompt.
5. The generated background is resized to the selected output format.
6. The product is cropped, resized, repositioned, and visually adjusted.
7. A blurred contact shadow is generated beneath the product.
8. The foreground and background layers are composited into the final image.

## 🛠️ Technologies

- Python
- Streamlit
- Pillow
- rembg
- U²-Net
- Hugging Face Inference Providers
- FLUX.1-schnell
- python-dotenv

## 📐 Output Formats

- Square Instagram post: 1080 × 1080
- Portrait Instagram post: 1080 × 1350
- Instagram Story or TikTok: 1080 × 1920
- Website landscape image: 1200 × 800

## 📁 Project Structure

```text
20_AI_Product_Photo_Studio/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

The `.env` file is excluded from GitHub because it contains the Hugging Face access token.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd 20_AI_Product_Photo_Studio
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Add the Hugging Face token

Create a `.env` file:

### 6. Run the application

```bash
streamlit run app.py
```

## 🧪 Real-World Testing

The application was tested using products with different visual properties:

- A rectangular product package with clearly defined edges
- A transparent perfume bottle with reflections and curved details

The system successfully preserved the products and placed them into newly generated scenes.

## ⚠️ Limitations

- Objects touching the product may be included in the foreground mask.
- Transparent and reflective products are more difficult to segment accurately.
- AI-generated backgrounds may not follow every prompt instruction.
- Product placement must sometimes be adjusted manually to match the surface.
- Background generation requires a Hugging Face token and available inference credits.
- Lighting adjustments improve the composition but do not perform full physical relighting.

## 🔮 Possible Future Improvements

- Interactive foreground-mask correction
- Automatic surface detection
- More advanced colour and lighting harmonisation
- Directional shadows based on scene lighting
- Batch product-photo processing

## 📌 Project Status

Core development completed. Deployment and final visual documentation are in progress.

## 👩‍💻 Author

**Shwe Yamin Oo**  
Data Science Graduate