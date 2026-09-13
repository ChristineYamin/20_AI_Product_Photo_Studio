import streamlit as st
from PIL import Image
from rembg import remove
from io import BytesIO
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image
from PIL import (
    Image,
    ImageOps,
    ImageDraw,
    ImageFilter,
    ImageEnhance,
)

load_dotenv()

def place_product_on_background(
    product_image,
    background_image,
    width_ratio,
    horizontal_ratio,
    vertical_ratio,
    brightness,
    contrast,
):
    alpha_channel = product_image.getchannel("A")
    product_box = alpha_channel.getbbox()

    if product_box is None:
        raise ValueError("No visible product was found.")

    cropped_product = product_image.crop(product_box)

    background_width, background_height = background_image.size

    target_width = int(background_width * width_ratio)
    target_height = int(
        cropped_product.height
        * target_width
        / cropped_product.width
    )

    maximum_height = int(background_height * 0.80)

    if target_height > maximum_height:
        target_height = maximum_height
        target_width = int(
            cropped_product.width
            * target_height
            / cropped_product.height
        )

    resized_product = cropped_product.resize(
        (target_width, target_height),
        Image.Resampling.LANCZOS,
    )

    product_alpha = resized_product.getchannel("A")
    product_rgb = resized_product.convert("RGB")

    product_rgb = ImageEnhance.Brightness(
        product_rgb
    ).enhance(brightness)

    product_rgb = ImageEnhance.Contrast(
        product_rgb
    ).enhance(contrast)

    resized_product = product_rgb.convert("RGBA")
    resized_product.putalpha(product_alpha)

    available_horizontal_space = background_width - target_width

    x_position = int(
        available_horizontal_space * horizontal_ratio
    )

    product_bottom = int(
        background_height * vertical_ratio
    )

    y_position = product_bottom - target_height

    x_position = max(
        0,
        min(x_position, background_width - target_width),
    )

    y_position = max(
        0,
        min(y_position, background_height - target_height),
    )

    product_bottom = y_position + target_height

    # Create a soft contact shadow
    shadow_layer = Image.new(
        "RGBA",
        background_image.size,
        (0, 0, 0, 0),
    )

    shadow_draw = ImageDraw.Draw(shadow_layer)

    shadow_width = int(target_width * 0.90)
    shadow_height = max(8, int(target_height * 0.04))

    shadow_left = (
        x_position + (target_width - shadow_width) // 2
    )

    shadow_top = product_bottom - shadow_height // 2

    shadow_draw.ellipse(
        (
            shadow_left,
            shadow_top,
            shadow_left + shadow_width,
            shadow_top + shadow_height,
        ),
        fill=(0, 0, 0, 110),
    )

    shadow_blur = max(6, int(target_width * 0.04))

    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(shadow_blur)
    )

    # Create the product layer
    product_layer = Image.new(
        "RGBA",
        background_image.size,
        (0, 0, 0, 0),
    )

    product_layer.paste(
        resized_product,
        (x_position, y_position),
        resized_product,
    )

    background_with_shadow = Image.alpha_composite(
        background_image,
        shadow_layer,
    )

    return Image.alpha_composite(
        background_with_shadow,
        product_layer,
    )


st.set_page_config(
    page_title="AI Product Photo Studio",
    page_icon="📸",
    layout="wide",
)

st.title("📸 AI Product Photo Studio")
st.write("Upload a product photo and transform its background.")

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    product_image = Image.open(uploaded_file).convert("RGBA")

    with st.spinner("Removing the background..."):
        transparent_image = remove(product_image)

    original_column, result_column = st.columns(2)

    with original_column:
        st.subheader("Original Image")
        st.image(product_image, width="stretch")

    with result_column:
        st.subheader("Background Removed")
        st.image(transparent_image, width="stretch")

    background_color = st.color_picker(
        "Choose a test background colour",
        "#F4D7D7",
    )

    rgb_color = tuple(
        int(background_color[i : i + 2], 16)
        for i in (1, 3, 5)
    )

    colour_background = Image.new(
        "RGBA",
        transparent_image.size,
        rgb_color + (255,),
    )

    colour_preview = Image.alpha_composite(
        colour_background,
        transparent_image,
    )

    st.subheader("Background Test")
    st.image(colour_preview, width="stretch")

    image_buffer = BytesIO()
    colour_preview.save(image_buffer, format="PNG")

    st.download_button(
        label="Download Product Image",
        data=image_buffer.getvalue(),
        file_name="product_photo.png",
        mime="image/png",
    )

    st.divider()
    st.subheader("Generate an AI Background")
                  

    background_prompt = st.text_area(
        "Describe the background",
        placeholder="A warm wooden café table with soft morning light",
    )

    if st.button("Generate Background"):
        if not background_prompt.strip():
            st.warning("Please describe the background first.")

        else:
            hf_token = os.getenv("HF_TOKEN")

            if not hf_token:
                st.error("Hugging Face token was not found.")

            else:
                try:
                    client = InferenceClient(
                        provider="auto",
                        api_key=hf_token,
                    )

                    with st.spinner("Generating your background..."):
                        generated_background = client.text_to_image(
                            background_prompt,
                            model="black-forest-labs/FLUX.1-schnell",
                        )

                    st.session_state.generated_background = (
                        generated_background
                    )

                except Exception as error:
                    st.error(
                        f"Background generation failed: {error}"
                    )

    if "generated_background" in st.session_state:
        generated_background = (
            st.session_state.generated_background
        )

        st.subheader("Generated Background")
        st.image(generated_background, width="stretch")

        st.subheader("Adjust Product Placement")

        product_size = st.slider(
            "Product size",
            min_value=15,
            max_value=60,
            value=25,
        )

        horizontal_position = st.slider(
            "Horizontal position",
            min_value=0,
            max_value=100,
            value=50,
        )

        vertical_position = st.slider(
            "Vertical position",
            min_value=20,
            max_value=100,
            value=70,
        )

        brightness = st.slider(
            "Product brightness",
            min_value=50,
            max_value=150,
            value=100,
        )

        contrast = st.slider(
            "Product contrast",
            min_value=50,
            max_value=150,
            value=100,
        )

        resized_background = ImageOps.fit(
            generated_background.convert("RGBA"),
            transparent_image.size,
            method=Image.Resampling.LANCZOS,
        )

        final_product_image = place_product_on_background(
            transparent_image,
            resized_background,
            width_ratio=product_size / 100,
            horizontal_ratio=horizontal_position / 100,
            vertical_ratio=vertical_position / 100,
            brightness=brightness / 100,
            contrast=contrast / 100,
        )

        st.subheader("Final Product Photo")
        st.image(final_product_image, width="stretch")

        final_image_buffer = BytesIO()

        final_product_image.convert("RGB").save(
            final_image_buffer,
            format="PNG",
        )

        st.download_button(
            label="Download Final Product Photo",
            data=final_image_buffer.getvalue(),
            file_name="ai_product_photo.png",
            mime="image/png",
        )