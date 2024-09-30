
# OCR Web Application <u>[Visit the App](https://huggingface.co/spaces/mannywho/webocr)</u>

This is a web application built using Streamlit that extracts text from images containing English and Hindi text. The application supports two OCR methods: **EasyOCR** and **Colpali-Qwen**. It allows users to upload an image, extract text, and search for keywords in the extracted text, highlighting them.

## Features

- **EasyOCR**: Real-time text extraction from images, supporting English and Hindi languages.
- **Colpali-Qwen**: Advanced text extraction using transformer-based models.
- **Keyword Search**: Search for and highlight keywords in the extracted text.
- **Device Adaptability**: Uses GPU if available for faster processing.

## Requirements

Before setting up the application, ensure you have the following dependencies installed:

- `Python 3.8+`
- `torch`
- `transformers`
- `easyocr`
- `streamlit`
- `Pillow`
- `numpy`

## Installation

To run the application locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ocr-web-app.git
cd ocr-web-app
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the Application Locally

```bash
git clone https://github.com/your-username/ocr-web-app.git
cd ocr-web-app
```

This will open the application in your default web browser. You can upload an image containing text and choose between EasyOCR or Colpali-Qwen for text extraction.


### 4. Deploy to Hugging Face Spaces

- To deploy the application on Hugging Face Spaces, follow these steps:
  - Create a new Hugging Face Space here.
  - In the "Create a Space" page, choose the "Streamlit" SDK.
  - Clone the newly created Space to your local machine:
 
```bash
git clone https://huggingface.co/spaces/your-username/ocr-web-app-space
cd ocr-web-app-space
```
Copy your application files to the Space repository.
Push the files to Hugging Face:
```bash
git add .
git commit -m "Initial commit"
git push
```
Your Streamlit app will be deployed and accessible through your Hugging Face Space.

## Using the Application<br><br>

### Upload an image containing Hindi or English text.<br>
<div style="margin-bottom: 20px;">
    <img src="https://github.com/user-attachments/assets/cc5c02b9-7913-45b0-b399-9d1436a93dc7" alt="image" width="600">
</div><br><br>

### Choose an OCR method from the sidebar (EasyOCR or Colpali-Qwen).<br>
<div style="margin-bottom: 20px;">
    <img src="https://github.com/user-attachments/assets/319bfb42-7fb6-4c73-b65e-1da4d53c24e4" alt="image" width="297">
</div><br><br>

### Once the text is extracted, enter a keyword to search within the extracted text.<br>
<div style="margin-bottom: 20px;">
    <img src="https://github.com/user-attachments/assets/a071a23f-1b4d-4ef4-a844-4926fbd216b3" alt="image" width="1227">
</div><br>
<div style="margin-bottom: 20px;">
    <img src="https://github.com/user-attachments/assets/e87d320e-72c2-4838-86f1-d76bdfbb246c" alt="image" width="1417">
</div><br><br>

### The application will highlight the keyword occurrences in the extracted text.<br>
<div style="margin-bottom: 20px;">
    <img src="https://github.com/user-attachments/assets/371c2b86-b5e2-481e-b6ff-19ef3ab76a67" alt="image" width="644">
</div><br>
