import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")

# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



#IMAGE FOR CHECK BOXES
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\CIMB AI LABS.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution

# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\fine 1 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 6.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution

# image_np = np.array(images[9])
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


# image_np = np.array(images[9])
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
# # #IMAGE FOR FIELDS


def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
    return black_pixel_ratio > 0.24

# def recognize_text_tess_digi(roi_image):
#     if is_blank_space(remove_lines(roi_image)):
#         return ""
#     custom_config = r'-c tessedit_char_whitelist=0123456789'
#     extracted_text=pytesseract.image_to_string(roi_image,config=custom_config)
#     return extracted_text.replace("\n","").strip()

# def recognize_text_tess_char(roi_image):
#     if is_blank_space(remove_lines(roi_image)):
#         return ""
#     name_config = r' -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     extracted_text=pytesseract.image_to_string(roi_image,config=name_config)
#     return extracted_text.replace("\n","").strip()


# def recognize_text_tess(roi_image):
#     # roi_image=remove_lines(roi_image)
#     if is_blank_space(remove_lines(roi_image)):
#         return ""
#     extracted_text = pytesseract.image_to_string(roi_image)
#     return extracted_text.replace("\n","").strip()

# def recognize_text_tess_2(roi_image):
#     return pytesseract.image_to_string(roi_image).replace("\n","").strip()

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


def islamic_credit_card(contours,img):
    result=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_img=img[y:y+h,x:x+w]
        
        if is_ticked(roi_img):
            result.append(recognize_text_trocr_2(img[y:y+h,x+w:x+w+350]))
    return result


def credit_card(contours,img):
    result=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_img=img[y:y+h,x:x+w]
        
        if is_ticked(roi_img):
            result.append(recognize_text_trocr_2(img[y:y+h,x+w:x+w+350]))
    return result

def card_type(contours,img):
    result=[]
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_img=img[y:y+h,x:x+w]
        
        if is_ticked(roi_img):
            result.append(recognize_text_trocr_2(img[y:y+h,x+w:x+w+300]))
    return result



# # Find contours of the CHECK BOXES
# contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_c, _ = cv2.findContours(img_bin_combined_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# ENTIRE CODE FOR CHECK BOX

def remove_lines_2(img):
    # print(img.shape)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    img_bin_c=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
    line_min_width = 55# Adjust based on your checkbox size
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
    return result


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
# # # print(checkbox_contours)
# checkbox_contours=remove_nearby_contours(checkbox_contours)

# field_contours=[]
# for i,c in enumerate(contours): 
#     area=cv2.contourArea(c)
#     if area>14000 and area<20000:

#         field_contours.append(c)

# extra=[]

# for c in contours:
#     area=cv2.contourArea(c)
#     if area>5500 and area<7000:
       
#         extra.append(c)

# for i,c in enumerate(extra):
#     x,y,w,h=cv2.boundingRect(c)
#     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

# image_np=cv2.resize(image_np,(550,850))
# cv2.imshow("Image",image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# field_contours=remove_nearby_contours(field_contours)

# result={}
# for i,c in enumerate(field_contours):
#     if i==0:
#         text,field=field_detection_2(c,img,image_np)
#     else:

#         text,field=field_detection(c,img,image_np)
#     if field in result.keys():
#         field=field+"2"
#     result[field]=text

# for i,c in enumerate(extra):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==2:
#         r=remove_lines_2(image_np[y:y+h,x+w+100:x+w+230])
#         # r=remove_lines(r)
#         # cv2.imshow("image",r)
#         result["CONVENTIONAL BRANCH CODE"]=recognize_text_trocr(r)

#     elif i==1:
#         r=remove_lines_2(image_np[y:y+h,x+w:x+w+240])
#         # r=remove_lines(r)
#         # cv2.imshow("Image",r)
#         result["EMPLOYEE NO."]=recognize_text_trocr(r)
#     else:
#         r=remove_lines(image_np[y:y+h,x+w+130:x+w+240])
#         # r=remove_lines(r)
#         # cv2.imshow("Image",r)
#         result["CAMPAIGN CODE"]=recognize_text_trocr(r)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()

# result["Islamic credit card:"]=islamic_credit_card(checkbox_contours[0:3],img)
# result["credit card:"]=credit_card(checkbox_contours[3:12],img)
# result["card type:"]=card_type(checkbox_contours[12:15],img)
# # print("Islamic credit card:",islamic_credit_card(checkbox_contours[0:3],img))
# # print("Credit card:",credit_card(checkbox_contours[3:12],img))
# # print("card type:",card_type(checkbox_contours[12:15],img))
# # # result=declaration(checkbox_contours[5:8],img)
# # if not result:
# #     print("Declaration : No")
# # else:
# #     print("Declaration: ",result)

# # total=field_contours+checkbox_contours+extra
# # for c in total:
# #     x,y,w,h=cv2.boundingRect(c)
# #     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
# # image_np=cv2.resize(image_np,(550,850))
# # cv2.imshow("Image",image_np)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()  

# df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
# print(df)
# df.to_excel('C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\results\\devanshi\\devanshi_10.xlsx',index = False)






  







# cv2.waitKey(0)
# cv2.destroyAllWindows()