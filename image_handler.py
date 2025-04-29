import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import zipfile
import pandas as pd
import json
import os

def encode_image(uploaded_file):
    """
    Encode an uploaded image to base64 string
    
    Args:
        uploaded_file: The uploaded file from st.file_uploader
        
    Returns:
        str: Base64 encoded string of the image
    """
    if uploaded_file is None:
        return None
    
    try:
        # Read the file and encode it to base64
        bytes_data = uploaded_file.getvalue()
        base64_encoded = base64.b64encode(bytes_data).decode()
        
        # Get file extension
        file_name = uploaded_file.name
        file_extension = file_name.split('.')[-1].lower()
        
        # Validate it's an allowed image format
        allowed_formats = ['jpg', 'jpeg', 'png', 'gif']
        if file_extension not in allowed_formats:
            st.error(f"Unsupported image format: {file_extension}. Please use jpg, jpeg, png, or gif.")
            return None
        
        return {
            'data': base64_encoded,
            'format': file_extension,
            'name': file_name
        }
    
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def decode_image(image_dict):
    """
    Decode a base64 encoded image dict
    
    Args:
        image_dict: Dictionary containing base64 encoded image data
        
    Returns:
        PIL.Image: Decoded image as PIL Image object
    """
    if image_dict is None or 'data' not in image_dict:
        return None
    
    try:
        # Decode base64 image
        base64_decoded = base64.b64decode(image_dict['data'])
        
        # Create PIL Image from bytes
        image = Image.open(BytesIO(base64_decoded))
        
        return image
    
    except Exception as e:
        st.error(f"Error decoding image: {str(e)}")
        return None

def display_image(image_dict):
    """
    Display a base64 encoded image in Streamlit
    
    Args:
        image_dict: Dictionary containing base64 encoded image data
    """
    if image_dict is None or 'data' not in image_dict:
        return
    
    try:
        # Create a data URL for displaying the image
        format_extension = image_dict.get('format', 'png')
        mime_type = f"image/{format_extension}"
        data_url = f"data:{mime_type};base64,{image_dict['data']}"
        
        # Display the image
        st.image(data_url, caption=image_dict.get('name', 'Image'))
    
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def export_with_images(trades_df, filename='trades_with_images.zip'):
    """
    Export trades with associated images as a zip file
    
    Args:
        trades_df: DataFrame containing trade data with image columns
        filename: Name of the zip file to create
        
    Returns:
        BytesIO: In-memory zip file
    """
    if trades_df.empty:
        return None
    
    # Create in-memory zip file
    zip_buffer = BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Export trades to CSV (excluding image data)
            trades_for_export = trades_df.copy()
            
            # Check if images column exists and create image references
            if 'images' in trades_for_export.columns:
                # Create a new column with image filenames instead of raw data
                trades_for_export['image_files'] = trades_for_export.apply(
                    lambda row: ','.join([f"{row.name}_{i}.{img.get('format', 'png')}" 
                                        for i, img in enumerate(row['images'])]) 
                                        if isinstance(row.get('images'), list) else '',
                    axis=1
                )
                # Remove the original images column with raw data
                trades_for_export = trades_for_export.drop(columns=['images'])
            
            # Save trades to CSV file
            trades_csv = trades_for_export.to_csv(index=False)
            zipf.writestr('trades.csv', trades_csv)
            
            # Save each image as a separate file
            if 'images' in trades_df.columns:
                for i, row in trades_df.iterrows():
                    if isinstance(row.get('images'), list):
                        for img_idx, img_dict in enumerate(row['images']):
                            if img_dict and 'data' in img_dict:
                                # Decode base64 data
                                img_data = base64.b64decode(img_dict['data'])
                                img_format = img_dict.get('format', 'png')
                                
                                # Save to zip with a unique name
                                img_filename = f"images/{i}_{img_idx}.{img_format}"
                                zipf.writestr(img_filename, img_data)
            
            # Create a manifest with metadata
            manifest = {
                'creation_date': pd.Timestamp.now().isoformat(),
                'num_trades': len(trades_df),
                'num_images': sum(len(row.get('images', [])) for _, row in trades_df.iterrows() 
                                 if isinstance(row.get('images'), list))
            }
            zipf.writestr('manifest.json', json.dumps(manifest, indent=2))
    
    except Exception as e:
        st.error(f"Error creating export file: {str(e)}")
        return None
    
    # Reset buffer position
    zip_buffer.seek(0)
    return zip_buffer

def import_with_images(uploaded_zip):
    """
    Import trades with associated images from a zip file
    
    Args:
        uploaded_zip: Uploaded zip file from st.file_uploader
        
    Returns:
        pd.DataFrame: DataFrame containing trade data with images
    """
    if uploaded_zip is None:
        return None
    
    try:
        # Create a BytesIO object from the uploaded file
        zip_buffer = BytesIO(uploaded_zip.getvalue())
        
        with zipfile.ZipFile(zip_buffer, 'r') as zipf:
            # Extract the trades CSV file
            if 'trades.csv' not in zipf.namelist():
                st.error("Invalid export file: 'trades.csv' not found.")
                return None
            
            # Read trades CSV
            trades_csv = zipf.read('trades.csv')
            trades_df = pd.read_csv(BytesIO(trades_csv))
            
            # Check if we have image references
            if 'image_files' in trades_df.columns:
                # Create a column for storing image data
                trades_df['images'] = None
                
                # Process each row
                for i, row in trades_df.iterrows():
                    if pd.notna(row['image_files']) and row['image_files']:
                        image_files = row['image_files'].split(',')
                        images_data = []
                        
                        for img_file in image_files:
                            img_file = img_file.strip()
                            if not img_file:
                                continue
                                
                            # Check if image exists in the zip
                            img_path = f"images/{img_file}"
                            if img_path in zipf.namelist():
                                img_data = zipf.read(img_path)
                                img_format = img_file.split('.')[-1].lower()
                                
                                # Encode to base64
                                base64_encoded = base64.b64encode(img_data).decode()
                                
                                # Add to images list
                                images_data.append({
                                    'data': base64_encoded,
                                    'format': img_format,
                                    'name': img_file
                                })
                        
                        # Store the image data list
                        trades_df.at[i, 'images'] = images_data
                
                # Remove the image_files column
                trades_df = trades_df.drop(columns=['image_files'])
            
            return trades_df
    
    except Exception as e:
        st.error(f"Error importing trades: {str(e)}")
        return None
