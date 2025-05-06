import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")


#IMAGE FOR CHECK BOXES
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\CIMB AI LABS.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution

# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 2 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 4.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# image_np = np.array(images[7])
# image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
# img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
# _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# img_bin=cv2.dilate(img_bin,np.ones((2,2),np.uint8),iterations=1)
# line_min_width = 15  # Adjust based on your checkbox size
# kernal_h = np.ones((1, line_min_width), np.uint8)
# kernal_v = np.ones((line_min_width, 1), np.uint8)
# img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
# img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
# img_bin_combined = cv2.add(img_bin_h, img_bin_v)

# image_np = np.array(images[7])
# image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
# img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
# _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# img_bin_c=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
# line_min_width = 15  # Adjust based on your checkbox size
# kernal_h = np.ones((1, line_min_width), np.uint8)
# kernal_v = np.ones((line_min_width, 1), np.uint8)
# img_bin_h = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_h)
# img_bin_v = cv2.morphologyEx(img_bin_c, cv2.MORPH_OPEN, kernal_v)
# img_bin_combined_c = cv2.add(img_bin_h, img_bin_v)

def is_blank_space(roi_image, white_pixel_threshold=0.98):
    # Convert the ROI to grayscale
    # gray_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to get a binary image
    _, binary_image = cv2.threshold(roi_image, 200, 255, cv2.THRESH_BINARY)
    
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

# def remove_nearby_contours(contours, min_distance=1):
#     filtered_contours = []
#     to_remove = set()
    
#     for i, contour1 in enumerate(contours):
#         if i in to_remove:
#             continue
        
#         x1, y1, w1, h1 = cv2.boundingRect(contour1)
#         keep = True
        
#         for j, contour2 in enumerate(contours):
#             if i != j and j not in to_remove:
#                 x2, y2, w2, h2 = cv2.boundingRect(contour2)
                
#                 # Calculate the distance between the centers of the bounding boxes
#                 center1 = np.array([x1 + w1 / 2, y1 + h1 / 2])
#                 center2 = np.array([x2 + w2 / 2, y2 + h2 / 2])
#                 distance = np.linalg.norm(center1 - center2)
#                 # print(distance, i, j)
                
#                 if distance <= min_distance:
#                     keep = False
#                     to_remove.add(j)
#                     break
        
#         if keep:
#             filtered_contours.append(contour1)
    
#     # Remove contours that are too close from the original list
#     print(to_remove)
#     contours = [c for i, c in enumerate(contours) if i not in to_remove]
#     # print(len(contours))
#     return contours
#IMAGE FOR FIELDS

def remove_lines(img):
    # print(img.shape)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    img_bin_c=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
    line_min_width = 150# Adjust based on your checkbox size
    kernal_h = np.ones((1, 70), np.uint8)
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
    return result

def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
    return black_pixel_ratio > 0.25

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


def declaration(contours,img):
    x, y, w, h = cv2.boundingRect(contours)
    roi_image = img[y:y+h, x:x+w]
        # result=[]
    if is_ticked(roi_image):
                return ("Yes, I would like to apply for Cash Plus Assurance ('CPA') which is underwritten by Sun Life Malaysia Assurance Berhad ('the Company'), an insurer registered with Bank Negara Malaysia under the Financial Services Act 2013, to cover the outstanding Cash Plus Personal loan amount in the event of death or total permanent disability.")
            
    return "NOTHING IS CHECKED"
          


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
    field_name = img[y:y+h, x-150:x]
    # print(i)
    try:
        # Check if field_name is empty
        if field_name is None or field_name.size == 0:
            raise ValueError("Field name region is empty or not loaded")
        
        

        text = recognize_text_trocr(image_np[y:y+h, x:x+w])
        field = recognize_text_trocr_2(field_name)
    
    except ValueError as ve:
        # print(f"Error: {ve}")
        text = recognize_text_trocr(img[y:y+h, x:x+w])
        field = recognize_text_trocr_2(img[y-30:y,x-40:x+w-100])  # Handle the empty case with a default value or message

    return text, field          



# Find contours of the CHECK BOXES
# contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_c, _ = cv2.findContours(img_bin_combined_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# ENTIRE CODE FOR CHECK BOX



# checkbox_contours = []
# field_contours=[]
# for i,contour in enumerate(contours_c):
#     x, y, w, h = cv2.boundingRect(contour)
#     aspect_ratio = w / float(h)
#     if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
#         # Filter based on size to exclude large or very small boxes
#         if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
#             area = cv2.contourArea(contour)
#             if area >250 and area <= 450:
#                 checkbox_contours.append(contour)
# # print(checkbox_contours)
# checkbox_contours=remove_nearby_contours(checkbox_contours)

# field_contours=[]
# for i,c in enumerate(contours):
#     area=cv2.contourArea(c)
#     if area>10000 and area<20000:
#         field_contours.append(c)

# result={}
# # x,y,w,h=cv2.boundingRect(field_contours[0])
# text,field=field_detection_2(field_contours[0],img)
# result[field]=text

# total_contours=field_contours+checkbox_contours

# for c in total_contours:
#     x,y,w,h=cv2.boundingRect(c)
#     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imwrite("pradyumn_8.png",image_np)

#         x,y,w,h=cv2.boundingRect(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#     # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#     # print(i,c)
# image_np=cv2.resize(image_np,(550,700))
# # cv2.imwrite("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\BB_images\\8th_page\\archer.png",image_np)
# cv2.imshow("Image",image_np)
# img_bin=cv2.resize(img_bin,(550,700))
# cv2.imshow("Binary Image",img_bin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# for i,c in enumerate(checkbox_contours):
#     x,y,w,h=cv2.boundingRect(c)
#     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#     cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#     # print(i,c)
# image_np=cv2.resize(image_np,(550,850))
# cv2.imshow("Third_page",image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# # contour=checkbox_contours[:5]
# # for i,c in enumerate(contour):
# #     x,y,w,h=cv2.boundingRect(c)
#     # print(recognize_text_trocr(image_np[y:y+h, x+w:x+w+200]))
#     # cv2.imshow("Feild",image_np[y:y+h, x+w:x+w+200])
# # # output_image=cv2.resize(output_image,(550,850))
# # # cv2.imwrite("detected_fields_page_1.png",output_image)
# # cv2.imshow("Detected feilds",output_image)


# result["declaration"]=declaration(checkbox_contours[0],img)

# df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
# # print(df)
# df.to_excel("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_8.xlsx")




