import streamlit as st
from PIL import Image
from rembg import remove
from io import BytesIO





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