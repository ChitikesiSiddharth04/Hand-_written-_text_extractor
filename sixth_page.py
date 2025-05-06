import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")


# #IMAGE FOR CHECK BOXES
# # images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\CIMB AI LABS.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\fine 1 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# # images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 5.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# image_np = np.array(images[5])
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


# image_np = np.array(images[5])
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




#IMAGE FOR FIELDS
def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    # cv2.imshow("Binary",binary)
    # cv2.waitKey(0)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
    return black_pixel_ratio > 0.4


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


def declaration(contours,img):
    
    result=[]
    for i,c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        
        if is_ticked(roi_image):
            if i==0:
                result.append("To the best of my knowledge, I have close relative(s) employed under the CIMB Group or who have acted as my guarantor.")
                
            else:
                result.append("I am the staff of the CIMB Group.")
                
    return result

            
    # return result
          
              
def acted_as_guarantor(contours,img):
    result=[]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        
        # Check if the region is ticked
        if is_ticked(roi_image):
            # # Debugging: Show the field image
            # cv2.imshow("Field", img[y:y+h, x+w:x+w+50])
            # cv2.waitKey(0)  # Wait for a key press to proceed (for debugging)
            
            # Recognize text next to the checkbox
            field = recognize_text_trocr_2(img[y:y+h, x+w:x+w+50])
            result.append(field)
            # return field 
    return result



# Find contours of the CHECK BOXES
# contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours_c, _ = cv2.findContours(img_bin_combined_c, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# ENTIRE CODE FOR CHECK BOX



# checkbox_contours = []
# field_contours=[]
# extra=[]
# for i,contour in enumerate(contours_c):
#     x, y, w, h = cv2.boundingRect(contour)
#     aspect_ratio = w / float(h)
#     if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
#         # Filter based on size to exclude large or very small boxes
#         if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
#             area = cv2.contourArea(contour)
#             if area >250 and area <= 450:
#                 checkbox_contours.append(contour)



# for i,c in enumerate(contours):
#     area=cv2.contourArea(c)
#     if area>5000 and area<9000:
#         extra.append(c)
#     # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#     # print(i,c)
# for c in contours:
#     area=cv2.contourArea(c)
#     if area>10000 and area<16000:
#         field_contours.append(c)

# field_contours=field_contours[::-1]

# # print(field_contours)
# total_contours=field_contours+extra+checkbox_contours
# for i,c in enumerate(total_contours):
#     x,y,w,h=cv2.boundingRect(c)
#     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#     # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
# # cv2.imwrite("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\pradyumn\\pradyumn_6.png",image_np)
# image_np=cv2.resize(image_np,(550,850))
# cv2.imshow("Image",image_np)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# result={}
# list1=field_contours[3:6]
# for i,c in enumerate(list1):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["Main_applicant_name_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     elif i==1:
#         result["Main_applicant_debt_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["Main_applicant_monthly_installment_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

# list2=field_contours[6:9]

# for i,c in enumerate(list2):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["Main_applicant_name_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     elif i==1:
#         result["Main_applicant_debt_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["Main_applicant_monthly_installment_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

# list3=field_contours[12:15]
# for i,c in enumerate(list3):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["Joint_applicant_name_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     elif i==1:
#         result["Joint_applicant_debt_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["Joint_applicant_monthly_installment_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

# list4=field_contours[15:18]

# for i,c in enumerate(list4):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["Joint_applicant_name_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     elif i==1:
#         result["Joint_applicant_debt_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["Joint_applicant_monthly_installment_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

# list5=[]
# # list5.append(field_contours[18])
# list5.append(field_contours[20])
# list5.append(field_contours[22])

# # list5=list5+extra[0:4]
# for i,c in enumerate(list5):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["CIMB_group_relative_name_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["CIMB_group_relative_name_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

# list6=[]
# list6.append(extra[3])
# list6.append(extra[1])

# for i,c in enumerate(list6):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["CIMB_group_relative_passport_no_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["CIMB_group_relative_passport_no_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])

# list7=[]
# list7.append(extra[2])
# list7.append(extra[0])


# for i,c in enumerate(list7):
#     x,y,w,h=cv2.boundingRect(c)
#     if i==0:
#         result["CIMB_group_relative_relationship_1"]=recognize_text_trocr(image_np[y:y+h,x:x+w])
#     else:
#         result["CIMB_group_relative_relationship_2"]=recognize_text_trocr(image_np[y:y+h,x:x+w])



# output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# contour=checkbox_contours[:5]
# for i,c in enumerate(contour):
#     x,y,w,h=cv2.boundingRect(c)
    # print(recognize_text_trocr(image_np[y:y+h, x+w:x+w+200]))
    # cv2.imshow("Feild",image_np[y:y+h, x+w:x+w+200])
# # output_image=cv2.resize(output_image,(550,850))
# # cv2.imwrite("detected_fields_page_1.png",output_image)
# cv2.imshow("Detected feilds",output_image)

    
# for c in checkbox_contours[0:4]:
#     x,y,w,h=cv2.boundingRect(c)
#     roi_img=img[y:y+h,x:x+w]
#     print(is_ticked(roi_img))

# result["acted_as_guarantor_1"]=acted_as_guarantor(checkbox_contours[2:4],image_np)
# result["acted_as_guarantor_2"]=acted_as_guarantor(checkbox_contours[0:2],image_np)

# r=declaration(checkbox_contours[4:6],img)
# if not result:
#     result["declaration"]="NOTHING IS CHECKED"
# else:
#     result["declaration"]=r

# df=pd.DataFrame(list(result.items()),columns=["Field","Text Extracted"])
# print(df)
# df.to_excel("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\MY_CODE\\selva\\selva_6.xlsx",index=False)







  







# cv2.waitKey(0)
# cv2.destroyAllWindows()