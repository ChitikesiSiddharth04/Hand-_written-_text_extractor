import cv2
import numpy as np
import time
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")

def remove_nearby_contours(contours, min_distance=1):
    filtered_contours = []
    to_remove = set()
    
    for i, contour1 in enumerate(contours):
        if i in to_remove:
            continue
        
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        keep = True
        
        for j, contour2 in enumerate(contours):
            if i != j and j not in to_remove:
                x2, y2, w2, h2 = cv2.boundingRect(contour2)
                
                # Calculate the distance between the centers of the bounding boxes
                center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
                center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
                distance = np.linalg.norm(center1 - center2)
                # print(distance, i, j)
                
                if distance <= min_distance:
                    keep = False
                    to_remove.add(j)
                    break
        
        if keep:
            filtered_contours.append(contour1)

    # print(to_remove)
    contours = [c for i, c in enumerate(contours) if i not in to_remove]
    return contours

def is_blank_space(roi_image, white_pixel_threshold=0.97):
    # Convert the ROI to grayscale
    # gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to get a binary image
    # _, binary_image = cv2.threshold(roi_image, 200, 255, cv2.THRESH_BINARY_INV)
    binary_image=roi_image
    # Calculate the total number of pixels in the region
    total_pixels = binary_image.size
    
    # Count the number of white pixels (255 value)
    white_pixels = np.sum(binary_image == 255)
    
    # Calculate the percentage of white pixels
    white_pixel_ratio = white_pixels / total_pixels

    # cv2.imshow("image",binary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(white_pixel_ratio)
    # If the percentage of white pixels exceeds the threshold, declare it as blank
    if white_pixel_ratio > white_pixel_threshold:
        return True  # The region is blank
    else:
        return False  # The region contains text or is not blank  
def remove_lines(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    img_bin_c=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
    line_min_width = 70# Adjust based on your checkbox size
    kernal_h = np.ones((1, line_min_width), np.uint8)
    kernal_v = np.ones((line_min_width, 1), np.uint8)
    img_bin_h = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_h)
    img_bin_v = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_v)
    img_bin_combined_c = cv2.add(img_bin_h, img_bin_v)
    # print(img_bin_combined_c.shape)
    img_bin_c=255-img_bin_c
    img_bin_c_bgr = cv2.cvtColor(img_bin_c, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("image",img_bin_c_bgr)
    img_bin_combined_c_bgr = cv2.cvtColor(img_bin_combined_c, cv2.COLOR_GRAY2BGR)
    # cv2.imshow("Image",img_bin_combined_c_bgr)
    result=cv2.bitwise_or(img_bin_c_bgr, img_bin_combined_c_bgr)
    # cv2.imshow("result",result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    result=cv2.erode(result,np.ones((1,1),np.uint8),iterations=50)
    return result
def recognize_text_trocr(roi_image):
    # roi_image=remove_lines(roi_image)
    if is_blank_space(remove_lines(roi_image)):
        return ""  # Skip OCR for blank regions
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def recognize_text_trocr_2(roi_image):

    # if is_blank_space(roi_image):
    #     return ""  # Skip OCR for blank regions
    pil_image = Image.fromarray(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB))
    pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def field_detection(contour,img,image_np):
    x,y,w,h=cv2.boundingRect(contour)
    field_name=img[y-30:y,x-50:x+w-100]
    # cv2.imshow("Field",field_name)
    # cv2.waitKey(0)
    text=recognize_text_trocr(image_np[y:y+h,x:x+w])
    field=recognize_text_trocr_2(field_name)
    return text,field

def field_detection_2(contour, img,image_np):
    x, y, w, h = cv2.boundingRect(contour)
    field_name = img[y:y+h, x-100:x]

    try:
        # Check if field_name is empty
        if field_name is None or field_name.size == 0:
            raise ValueError("Field name region is empty or not loaded")
        
        text = recognize_text_trocr(image_np[y:y+h, x:x+w])
        field = recognize_text_trocr_2(field_name)
    
    except ValueError as ve:
        # print(f"Error: {ve}")
        text = recognize_text_trocr(image_np[y:y+h, x:x+w])
        field = recognize_text_trocr_2(img[y-30:y,x-40:x+w-100])  # Handle the empty case with a default value or message

    return text, field