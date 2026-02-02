from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from bidi.algorithm import get_display
import arabic_reshaper

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

def fix_arabic_text(text):
    reshaped_text = arabic_reshaper.reshape(text)
    return get_display(reshaped_text)

# Configure OCR for Arabic and English
ocr_options = TesseractCliOcrOptions(
    lang=["ara", "eng"],
    force_full_page_ocr=True
)

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    ocr_options=ocr_options,
    images_scale=5.0,
    do_table_structure=True
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

doc = converter.convert("organized.pdf").document
extracted_text = doc.export_to_markdown()
#fixed_text = fix_arabic_text(extracted_text)

# Save the processed text to a local file
with open("organized_output.md", "w", encoding="utf-8") as f:
    f.write(extracted_text)
