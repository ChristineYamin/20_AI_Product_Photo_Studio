import hashlib
import os
from io import BytesIO
import torch
from transformers import (
    Sam2Model,
    Sam2Processor,
)
from streamlit_image_coordinates import (
    streamlit_image_coordinates,
)
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)


load_dotenv()
SAM2_MODEL_NAME = "facebook/sam2.1-hiera-tiny"



@st.cache_resource
def load_sam2_model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    token = os.getenv("HF_TOKEN")

    processor = Sam2Processor.from_pretrained(
        SAM2_MODEL_NAME,
        token=token,
    )

    model = Sam2Model.from_pretrained(
        SAM2_MODEL_NAME,
        token=token,
    )

    model.to(device)
    model.eval()

    return processor, model, device
@st.cache_data(show_spinner=False)
def remove_background_with_sam2(
    image_bytes,
    selected_points,
):
    processor, model, device = load_sam2_model()

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

    # Treat every click as a separate prompt.
    point_prompts = [
        [[int(x), int(y)]]
        for x, y in selected_points
    ]

    label_prompts = [
        [1]
        for _ in selected_points
    ]

    inputs = processor(
        images=image,
        input_points=[point_prompts],
        input_labels=[label_prompts],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            multimask_output=True,
        )

    processed_masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
    )[0]

    quality_scores = outputs.iou_scores[0].cpu()

    combined_mask = torch.zeros_like(
        processed_masks[0, 0],
        dtype=torch.bool,
    )

    # Select the best mask for each click, then combine them.
    for point_index in range(
        len(selected_points)
    ):
        best_mask_index = int(
            torch.argmax(
                quality_scores[point_index]
            )
        )

        point_mask = processed_masks[
            point_index,
            best_mask_index,
        ] > 0

        combined_mask = (
            combined_mask | point_mask
        )

    mask_array = (
        combined_mask
        .to(torch.uint8)
        .numpy()
        * 255
    )

    mask_image = Image.fromarray(
        mask_array
    ).convert("L")

    mask_image = mask_image.filter(
        ImageFilter.MedianFilter(size=5)
    )

    # Fill enclosed transparent holes, such as the
    # inside of a bowl or transparent bottle.
    inverted_mask = ImageOps.invert(mask_image)
    enclosed_holes = inverted_mask.copy()

    ImageDraw.floodfill(
        enclosed_holes,
        xy=(0, 0),
        value=0,
    )

    mask_image = ImageChops.lighter(
        mask_image,
        enclosed_holes,
    )

    mask_image = mask_image.filter(
        ImageFilter.GaussianBlur(radius=1)
    )

    transparent_product = image.convert("RGBA")
    transparent_product.putalpha(mask_image)

    return transparent_product

def image_to_bytes(image):
    """Convert a Pillow image into downloadable PNG bytes."""
    image_buffer = BytesIO()
    image.convert("RGB").save(image_buffer, format="PNG")
    return image_buffer.getvalue()


def place_product_on_background(
    product_image,
    background_image,
    width_ratio,
    horizontal_ratio,
    vertical_ratio,
    brightness,
    contrast,
):
    """Crop, resize, adjust and position a product on a background."""
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

    # Adjust product lighting
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

    # Calculate product position
    available_horizontal_space = (
        background_width - target_width
    )

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

    # Create contact shadow
    shadow_layer = Image.new(
        "RGBA",
        background_image.size,
        (0, 0, 0, 0),
    )

    shadow_draw = ImageDraw.Draw(shadow_layer)

    shadow_width = int(target_width * 0.90)
    shadow_height = max(
        8,
        int(target_height * 0.04),
    )

    shadow_left = (
        x_position
        + (target_width - shadow_width) // 2
    )

    shadow_top = (
        product_bottom - shadow_height // 2
    )

    shadow_draw.ellipse(
        (
            shadow_left,
            shadow_top,
            shadow_left + shadow_width,
            shadow_top + shadow_height,
        ),
        fill=(0, 0, 0, 110),
    )

    shadow_blur = max(
        6,
        int(target_width * 0.04),
    )

    shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(shadow_blur)
    )

    # Create product layer
    product_layer = Image.new(
        "RGBA",
        background_image.size,
        (0, 0, 0, 0),
    )

    product_layer.paste(
        resized_product,
        (x_position, y_position),
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
st.write(
    "Remove a product's background and create "
    "a polished promotional image."
)

uploaded_file = st.file_uploader(
    "Upload a product image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload a product image to begin.")
    st.stop()


try:
    uploaded_bytes = uploaded_file.getvalue()
    product_image = Image.open(
        BytesIO(uploaded_bytes)
    ).convert("RGBA")

except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()


# Clear the old AI background when a new product is uploaded
file_signature = hashlib.md5(
    uploaded_bytes
).hexdigest()

if (
    st.session_state.get("uploaded_signature")
    != file_signature
):
    
    
    st.session_state.uploaded_signature = file_signature


    st.session_state.pop(
        "generated_background",
        None,
    )

    st.session_state.pop("sam_points", None)
    st.session_state.pop("last_sam_click", None)

st.subheader("Select the Product")

st.info(
    "Click at least 2 areas inside the product. "
    "Add more points if any part is missing."
)

if "sam_points" not in st.session_state:
    st.session_state.sam_points = []

if "sam_click_version" not in st.session_state:
    st.session_state.sam_click_version = 0

if st.button("Clear Selection Points"):
    st.session_state.sam_points = []
    st.session_state.sam_click_version += 1
    st.session_state.pop(
        "last_sam_click",
        None,
    )
    st.rerun()

selection_image = product_image.convert("RGB").copy()

selection_image.thumbnail(
    (700, 700),
    Image.Resampling.LANCZOS,
)

# Draw every selected point on the displayed image.
point_draw = ImageDraw.Draw(selection_image)

point_radius = max(
    7,
    min(selection_image.size) // 70,
)

for point_number, (original_x, original_y) in enumerate(
    st.session_state.sam_points,
    start=1,
):
    display_x = int(
        original_x
        * selection_image.width
        / product_image.width
    )

    display_y = int(
        original_y
        * selection_image.height
        / product_image.height
    )

    point_draw.ellipse(
        (
            display_x - point_radius,
            display_y - point_radius,
            display_x + point_radius,
            display_y + point_radius,
        ),
        fill="#FF3B30",
        outline="white",
        width=3,
    )

    point_draw.text(
        (
            display_x - 4,
            display_y - 7,
        ),
        str(point_number),
        fill="white",
    )

click_result = streamlit_image_coordinates(
    selection_image,
    key=(
        f"sam_click_{file_signature}_"
        f"{st.session_state.sam_click_version}"
    ),
)

if click_result is not None:
    click_signature = click_result.get(
        "unix_time",
        (
            click_result["x"],
            click_result["y"],
        ),
    )

    if (
        st.session_state.get("last_sam_click")
        != click_signature
    ):
        original_x = int(
            click_result["x"]
            * product_image.width
            / selection_image.width
        )

        original_y = int(
            click_result["y"]
            * product_image.height
            / selection_image.height
        )

        st.session_state.sam_points.append(
            (original_x, original_y)
        )

        st.session_state.last_sam_click = (
            click_signature
        )

        # Refresh so the new numbered dot appears immediately.
        st.rerun()

number_of_points = len(
    st.session_state.sam_points
)

if number_of_points < 2:
    st.warning(
        f"{number_of_points}/2 points selected. "
        "Please click another area of the product."
    )
    st.stop()

st.success(
    f"{number_of_points} selection points added."
)

with st.spinner(
    "SAM 2 is selecting your product..."
):
    transparent_image = (
        remove_background_with_sam2(
            uploaded_bytes,
            tuple(st.session_state.sam_points),
        )
    )

with st.expander("View image preparation", expanded=False):
    original_column, removed_column = st.columns(2)

    with original_column:
        st.subheader("Original")
        st.image(
            product_image,
            width="stretch",
        )

    with removed_column:
        st.subheader("Background Removed")
        st.image(
            transparent_image,
            width="stretch",
        )


solid_tab, ai_tab = st.tabs(
    [
        "🎨 Solid Colour",
        "✨ AI Background",
    ]
)


with solid_tab:
    st.subheader("Solid Colour Background")

    background_color = st.color_picker(
        "Choose a background colour",
        "#F4D7D7",
    )

    rgb_color = tuple(
        int(background_color[i:i + 2], 16)
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

    st.image(
        colour_preview,
        width="stretch",
    )

    st.download_button(
        label="Download Solid Colour Photo",
        data=image_to_bytes(colour_preview),
        file_name="solid_colour_product.png",
        mime="image/png",
    )


with ai_tab:
    st.subheader("Generate an AI Background")

    product_type = st.selectbox(
        "Product type",
        [
            "Perfume",
            "Cosmetics",
            "Jewellery",
            "Food",
            "Electronics",
            "Other",
        ],
    )
    product_view = st.radio(
        "Product view",
        options=[
            "Upright",
            "Top-down",
        ],
        index=(
            1
            if product_type == "Food"
            else 0
        ),
        horizontal=True,
    )
    background_style = st.selectbox(
    "Background style",
    [
        "Luxury",
        "Studio",
        "Cinematic",
        "Showroom",
        "Nature",
        "Café",
    ],
)

    product_descriptions = {
        "Perfume": "elegant amber and cream lighting",
        "Cosmetics": "soft pastel colours and diffused lighting",
        "Jewellery": "deep velvet tones and precise luxury lighting",
        "Food": "warm appetising colours and natural lighting",
        "Electronics": "cool modern colours and crisp lighting",
        "Other": "balanced neutral colours and professional lighting",
    }

    upright_styles = {
        "Luxury": (
            "an elegant luxury room with a broad marble surface "
            "in the foreground"
        ),
        "Studio": (
            "a minimal photography studio with a matte display surface"
        ),
        "Cinematic": (
            "a dramatic cinematic room with a dark display surface"
        ),
        "Showroom": (
            "a modern high-end showroom with a clean display counter"
        ),
        "Nature": (
            "a soft natural setting with a stone or wooden surface"
        ),
        "Café": (
            "a warm stylish café interior with a clear wooden table "
            "in the foreground"
        ),
    }

    top_down_styles = {
        "Luxury": (
            "one continuous polished marble surface with subtle "
            "gold-toned lighting around the edges"
        ),
        "Studio": (
            "one continuous neutral matte studio surface with soft "
            "diffused lighting"
        ),
        "Cinematic": (
            "one continuous dark textured surface with dramatic "
            "diagonal light and soft shadows"
        ),
        "Showroom": (
            "one continuous polished minimal surface with clean "
            "architectural lighting"
        ),
        "Nature": (
            "one continuous natural stone or wooden surface with "
            "subtle leaves around the outer edges"
        ),
        "Café": (
            "one continuous warm wooden café tabletop with natural "
            "wood grain and soft window light"
        ),
    }

    if product_view == "Top-down":
        selected_style = top_down_styles[background_style]

        composition_description = (
            "Strict overhead flat-lay photography. "
            "The camera is directly above the surface at exactly "
            "90 degrees, pointing straight down. "
            "The entire image is a single flat horizontal surface. "
            "No room, wall, horizon, furniture, chairs, windows, "
            "shelves, podium, pedestal, plate, bowl, food, bottle, "
            "product, or vertical objects."
        )

    else:
        selected_style = upright_styles[background_style]

        composition_description = (
            "Front-facing commercial product photography. "
            "Show a broad horizontal surface in the foreground "
            "with realistic perspective."
        )

    if product_view == "Top-down":
        background_prompt = (
            "EMPTY WOODEN TABLETOP TEXTURE. "
            "Strict 90-degree overhead photograph with the camera "
            "pointing directly downward. "
            f"{selected_style}. "
            "The entire image must contain only one continuous flat "
            "surface with an empty centre. "
            "No food, bowl, plate, cup, product, cutting board, props, "
            "chairs, windows, walls, room, horizon, furniture, text, "
            "logo, or watermark."
        )

    else:
        background_prompt = (
            "Create an empty photorealistic commercial background "
            "plate only. "
            f"Use {product_descriptions[product_type]}. "
            f"Scene: {selected_style}. "
            f"{composition_description} "
            "Keep the centre completely empty for adding a product later. "
            "Do not generate the product itself. "
            "No text, logo, watermark, people, or extra display platform."
        )

    


    

    

    
       
     
        
    

    

    
    with st.expander("View generated prompt"):
        st.write(background_prompt)

    

    if st.button(
        "Generate Background",
        type="primary",
    ):
        if not background_prompt.strip():
            st.warning(
                "Please describe the background first."
            )

        else:
            hf_token = os.getenv("HF_TOKEN")

            if not hf_token:
                st.error(
                    "Hugging Face token was not found."
                )

            else:
                try:
                    client = InferenceClient(
                        provider="auto",
                        api_key=hf_token,
                    )

                    with st.spinner(
                        "Generating your background..."
                    ):
                        generated_background = (
                            client.text_to_image(
                                background_prompt,
                                model=(
                                    "black-forest-labs/"
                                    "FLUX.1-schnell"
                                ),
                            )
                        )

                    st.session_state.generated_background = (
                        generated_background
                    )

                except Exception as error:
                    st.error(
                        "Background generation failed: "
                        f"{error}"
                    )

    if "generated_background" in st.session_state:
        if st.button("Clear AI Background"):
            del st.session_state.generated_background
            st.rerun()
    if "generated_background" in st.session_state:
        generated_background = (
            st.session_state.generated_background
        )

        output_formats = {
            "Square — Instagram (1080 × 1080)": (
                1080,
                1080,
            ),
            "Portrait — Instagram (1080 × 1350)": (
                1080,
                1350,
            ),
            "Story — Instagram/TikTok (1080 × 1920)": (
                1080,
                1920,
            ),
            "Landscape — Website (1200 × 800)": (
                1200,
                800,
            ),
        }

        st.subheader("Final Product Photo")
        final_photo_placeholder = st.empty()

        with st.expander(
            "🎛️ Adjust Product",
            expanded=True,
        ):

            selected_format = st.selectbox(
                "Output format",
                options=list(output_formats.keys()),
            )

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
                key="horizontal_position",
            )

            vertical_position = st.slider(
                "Vertical position",
                min_value=20,
                max_value=100,
                value=70,
                key="vertical_position",
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

        output_size = output_formats[
            selected_format
        ]

        resized_background = ImageOps.fit(
            generated_background.convert("RGBA"),
            output_size,
            method=Image.Resampling.LANCZOS,
        )

        final_product_image = (
            place_product_on_background(
                transparent_image,
                resized_background,
                width_ratio=product_size / 100,
                horizontal_ratio=(
                    horizontal_position / 100
                ),
                vertical_ratio=(
                    vertical_position / 100
                ),
                brightness=brightness / 100,
                contrast=contrast / 100,
            )
        )

        final_photo_placeholder.image(
            final_product_image,
            width="stretch",
        )

        st.caption(
            f"Output size: {output_size[0]} × "
            f"{output_size[1]} pixels"
        )

        st.download_button(
            label="Download Final Product Photo",
            data=image_to_bytes(final_product_image),
            file_name="ai_product_photo.png",
            mime="image/png",
        )

        with st.expander(
            "View generated background"
        ):
            st.image(
                generated_background,
                width="stretch",
            )