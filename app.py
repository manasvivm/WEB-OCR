import streamlit as st
from PIL import Image, ImageOps
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from byaldi import RAGMultiModalModel
import easyocr
import numpy as np
import os

# Set the device to GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 


# Function to generate unique document IDs for Colpali indexing
if 'doc_id_counter' not in st.session_state:
    st.session_state.doc_id_counter = 0



def generate_doc_id():
    st.session_state.doc_id_counter += 1
    return st.session_state.doc_id_counter


# Cache model and processor for Colpali-Qwen
@st.cache_resource
def load_model():
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)
    # RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")
    RAG= None
    return model, processor, RAG


model, processor, RAG = load_model()

# Sidebar for OCR methods
st.sidebar.header("Choose OCR Method")
ocr_method = st.sidebar.radio(
    "Select an OCR method:",
    ["EasyOCR", "Colpali-Qwen (may take time to load results due to low resources)"],
    index=0  # Set EasyOCR as the default selection
)

st.title("OCR for Hindi and English Text")
st.write("Upload an image containing Hindi or English text to extract it using the selected OCR method.")

# Common image uploader
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
st.session_state.colpali_indexed = False



if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""



# OCR using EasyOCR
if ocr_method.startswith("EasyOCR") and uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image (EasyOCR)", use_column_width=False, width=300)  # Display a smaller image

    # Use GPU for EasyOCR if available
    reader = easyocr.Reader(['en', 'hi'], gpu=torch.cuda.is_available())
    image_np = np.array(image)  # Convert image to numpy array for OCR processing
    extracted_text = reader.readtext(image_np, detail=0)  # Extract text
    st.session_state.extracted_text = " ".join(extracted_text)
    st.session_state.colpali_indexed = False



# OCR using Colpali-Qwen
elif ocr_method.startswith("Colpali-Qwen") and uploaded_image is not None:
    # Avoid reindexing by checking if the image has already been processed
    if 'colpali_indexed' not in st.session_state or not st.session_state.colpali_indexed:
        
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image (Colpali-Qwen)", use_column_width=False, width=300)
        image = ImageOps.grayscale(image)  # Convert image to grayscale for better OCR performance
        image = image.resize((800, 800))  # Resize image to optimize processing

        doc_id = generate_doc_id()
          # Display a smaller image

        # Save temporary file for RAG indexing
        temp_image_path = "/tmp/temp_image.png"
        image.save(temp_image_path)

        # Index the image with a unique doc_id - for multiple images 
        # RAG.index(
        #     input_path=temp_image_path,
        #     index_name="image_index",
        #     doc_ids=[doc_id],
        #     store_collection_with_index=False,
        #     overwrite=True
        # )
        os.remove(temp_image_path)

        # Save index flag in session state to avoid reindexing
        st.session_state.colpali_indexed = True

        # Now generate text only once
        text_query = "Extract all text - English or hindi from image"
        # results = RAG.search(text_query, k=50)
        # print(results)

        # Prepare input for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_query}
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        # Generate output using the model
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        st.session_state.extracted_text = output_text

# Common keyword search functionality
if st.session_state.extracted_text:
    st.subheader("Extracted Text:")
    st.write(st.session_state.extracted_text)
    
    keyword = st.text_input("Enter a keyword to search:")

    if keyword:
        if isinstance(st.session_state.extracted_text, list):
            extracted_text_str = " ".join(st.session_state.extracted_text)
        else:
            extracted_text_str = st.session_state.extracted_text

        highlighted_text = extracted_text_str.replace(
            keyword, f"<span style='background-color: yellow; color: black; font-weight: bold;'>{keyword}</span>"
        )
        st.session_state.colpali_indexed = False
        st.markdown(f"**Highlighted Text:** {highlighted_text}", unsafe_allow_html=True)