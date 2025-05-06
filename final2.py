import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# import matplotlib.pyplot as plt

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")


#IMAGE FOR CHECK BOXES
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\CIMB AI LABS.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# # images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 2 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:/Users/DELL/Downloads/File - 005.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:/Users/DELL/Downloads/poppler-24.07.0/Library/bin")  # Set dpi for higher resolution
# image_np = np.array(images[1])
# image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
# img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
# _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# img_bin=cv2.dilate(img_bin,np.ones((3,3),np.uint8),iterations=1)
# line_min_width = 15  # Adjust based on your checkbox size
# kernal_h = np.ones((1, line_min_width), np.uint8)
# kernal_v = np.ones((line_min_width, 1), np.uint8)
# img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
# img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
# img_bin_combined = cv2.add(img_bin_h, img_bin_v)

# _, img_bin_cb = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# img_bin_cb =cv2.dilate(img_bin_cb,np.ones((1,1),np.uint8),iterations=1)
# line_min_width = 15  # Adjust based on your checkbox size
# kernal_h = np.ones((1, line_min_width), np.uint8)
# kernal_v = np.ones((line_min_width, 1), np.uint8)
# img_bin_h_cb = cv2.morphologyEx(img_bin_cb, cv2.MORPH_OPEN, kernal_h)
# img_bin_v_cb = cv2.morphologyEx(img_bin_cb, cv2.MORPH_OPEN, kernal_v)
# img_for_checkbox = cv2.add(img_bin_h_cb, img_bin_v_cb)

# contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_cb, _ = cv2.findContours(img_for_checkbox, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

#     # print(to_remove)
#     contours = [c for i, c in enumerate(contours) if i not in to_remove]
#     return contours

def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    return black_pixel_ratio > 0.25

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

def size_of_current_employment(contours,img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        if is_ticked(roi_image):
          return recognize_text_trocr_2(img[y:y+h,x+w:x+w+200])
    return "NOTHING IS CHECKED"

def type_of_company(contours, img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        if is_ticked(roi_image):
          
            field = recognize_text_trocr_2(img[y:y+h, x+w:x+w+300])
            if "other" not in field.strip().lower():
                return field
            else:
                return recognize_text_trocr(img[y:y+h, x+w+80:x+w+550])
    
    return "NOTHING IS TICKED"



def employee_status(contours,img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        if is_ticked(roi_image):
          return recognize_text_trocr_2(img[y:y+h,x+w:x+w+200])
    return "NOTHING IS CHECKED"

def preferred_mailing_address(contours, img):
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        if is_ticked(roi_image):
            text_roi = img[y:y+h, x+w:x+w+200]
            return recognize_text_trocr_2(text_roi)
    
    return "NOTHING IS CHECKED"

def residence_type(contours,img):
    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        roi_image=img[y:y+h,x:x+w]
        if is_ticked(roi_image):
            return recognize_text_trocr_2(img[y:y+h,x+w:x+w+220])
    return "NOTHING IS CHECKED"

# contour_cb = remove_nearby_contours(contours_cb)

# cb_c_size = []
# cb_c_type = []
# cb_c_employement = []
# cb_c_mailing = []
# cb_c_residence = []
# field_contours=[]
# for i,c in enumerate(contours_cb):
#     x, y, w, h = cv2.boundingRect(c)
#     aspect_ratio = w / float(h)
#     if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
#         # Filter based on size to exclude large or very small boxes
#         if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
#             area = cv2.contourArea(c)
#             if area >250 and area <= 450 and y > 329 and y < 495:
#                 cb_c_residence.append(c)
#             if area >250 and area <= 450 and y > 157 and y < 226 and x > 700:
#                 cb_c_mailing.append(c)
#             if area >250 and area <= 450 and y > 780 and y < 870 and x>700:
#                 cb_c_employement.append(c)
#             if area >250 and area <= 450 and y > 935 and y < 1215 and x>700:
#                 cb_c_type.append(c)
#             if area >250 and area <= 450 and y > 1215 and x>700:
#                 cb_c_size.append(c)

#                 # cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)

# # contours_needed = [c for i, c in enumerate(contours_cb) if i in checkbox_contours]
# # # contour=checkbox_contours[42:48]
# # for i,c in enumerate(contours_needed):
# #     x,y,w,h=cv2.boundingRect(c)
# #     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
# #     cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#     # print(recognize_text_trocr(image_np[y:y+h,x+w:x+w+300]))
#     # cv2.imshow("Feild",image_np[y:y+h,x+w:x+w+300])

# field_contours = []
# field_contours_sl = []
# contours_keep = remove_nearby_contours(contours)
# for i,c in enumerate(contours_keep):
#     area=cv2.contourArea(c)
#     x,y,w,h=cv2.boundingRect(c)
#     if area>10000 and area<20000:
#         x,y,w,h=cv2.boundingRect(c)
#         field_contours.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#     elif area > 2500 and area < 5000 and y > 1000 and x > 460 and x < 730:
#         field_contours_sl.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
    
#     elif area > 2500 and area < 5000 and y > 1000 and x >= 660:
#         field_contours.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)

    


#     # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#     # print(i,c)
# result = {}

# for c in field_contours:
#     text,field=field_detection(c,img)
#     if field in result.keys():
#         field=field+"2"
#     result[field]=text

# for c in field_contours_sl:
#     text,field=field_detection_2(c,img)
#     if field in result.keys():
#         field=field+"2"
#     result[field]=text

# result["Size of current employment"] = size_of_current_employment(cb_c_size, img)
# result["Type of company"] = type_of_company(cb_c_type ,img)
# result["Employee Status"] = employee_status(cb_c_employement,img)
# result["preferred_mailing_address"] = preferred_mailing_address(cb_c_mailing,img)
# result["Residence Type"] = residence_type(cb_c_residence,img)

# print(result)
# order_text = dict(sorted(result.items(),reverse = True))

# df = pd.DataFrame(list(order_text.items()),columns = ['Field','Extracted Text' ])

# df.to_csv('aditya2.csv',index = False)

# # image _np=cv2.resize(image_np,(550,700))
# # cv2.imwrite("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\BB_images\\2nd_page\\cimb.png",image_np)
# plt.imshow(image_np,cmap = 'gray')
# plt.show()
# cv2.imshow("Image",image_np)
# img_bin=cv2.resize(img_bin,(550,700))
# cv2.imshow("Binary Image",img_bin)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# contour=checkbox_contours[23:25]
# for i,c in enumerate(contour):
#     x,y,w,h=cv2.boundingRect(c)
#     print(recognize_text_trocr(image_np[y:y+h, x+w:x+w+200]))
#     cv2.imshow("Feild",image_np[y:y+h, x+w:x+w+200])
# # # output_image=cv2.resize(output_image,(550,850))
# # # cv2.imwrite("detected_fields_page_1.png",output_image)
# # cv2.imshow("Detected feilds",output_image)

# print("Size of current employment: ",size_of_current_employment(checkbox_contours[1:4],img))
# print("Type of company: ",type_of_company(checkbox_contours[4:14],img))
# print("Employee Status: ",employee_status(checkbox_contours[15:18],img))
# print("preferred_mailing_address: ",preferred_mailing_address(checkbox_contours[23:25],img))
# print("Residence Type: ",residence_type(checkbox_contours[18:23],img))









  






