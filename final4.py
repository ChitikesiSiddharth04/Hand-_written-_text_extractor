import cv2
import numpy as np
from pdf2image import convert_from_path
import pandas as pd
# import matplotlib.pyplot as plt
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_processor")
model = VisionEncoderDecoderModel.from_pretrained("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\AJK_CODE\\EXECUTABLE\\trocr_model")


# #IMAGE FOR CHECK BOXES
# images = convert_from_path("C:/Users/DELL/Downloads/File - 004.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:/Users/DELL/Downloads/poppler-24.07.0/Library/bin")  # Set dpi for higher resolution
# # images = convert_from_path("C:/Users/DELL/Downloads/File - 004.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:/Users/DELL/Downloads/poppler-24.07.0/Library/bin")  # Set dpi for higher resolution
# image_np = np.array(images[3])
# image_np=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
# img=cv2.cvtColor(image_np,cv2.COLOR_RGB2GRAY)
# _, img_bin = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
# img_bin=cv2.dilate(img_bin,np.ones((1,1),np.uint8),iterations=1)
# line_min_width = 15  # Adjust based on your checkbox size
# kernal_h = np.ones((1, line_min_width), np.uint8)
# kernal_v = np.ones((line_min_width, 1), np.uint8)
# img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
# img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
# img_bin_combined = cv2.add(img_bin_h, img_bin_v)

# pa = []
# ja = []
# ja_2 = []
# pep_pa = []
# pep_ja1 = []
# pep_ja2 = []
# pep_fm_pa = []
# pep_fm_ja1 = []
# pep_fm_ja2 = []
# pep_ca_pa = []
# pep_ca_ja1 = []
# pep_ca_ja2 = []
# fm_pa_code = []
# fm_ja1_code = []
# fm_ja2_code = []
# ca_pa_code = []
# ca_ja1_code = []
# ca_ja2_code = []
# others_pa = []
# others_ja1 = []
# others_ja2 = []

# text = {}
#IMAGE FOR FIELDS
def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
    return black_pixel_ratio > 0.35

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

def yes_no(contours,img):
    result = []
    for i,c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        image = img[y:y+h, x:x+w]
        if is_ticked(image):
            # cv2.rectangle(img,(x+w,y),(x+w+50,y+h),(0,255,0),3)
            # if "yes" in recognize_text_trocr(img[y:y+h,x+w:x+w+50]).strip().lower():
            result.append(recognize_text_trocr(img[y:y+h,x+w:x+w+50]).strip().lower())
    return result
def pep(contours,img):
    for i,c in enumerate(contours):
        x,y,w,h = cv2.boundingRect(c)
        image = img[y:y+h, x:x+w]
        if is_ticked(image):
            return 'Yes'
    return 'No'

def code(contours,img):
    result = []
    for i,c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        # if white_pix(roi_image):
        result.append(recognize_text_trocr(roi_image))
        # else:
        #     result = " "
    return result

# contours, _ = cv2.findContours(img_bin_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = remove_nearby_contours(contours)
# for i,contour in enumerate(contours):
#     x, y, w, h = cv2.boundingRect(contour)
#     aspect_ratio = w / float(h)
#     if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
#         # Filter based on size to exclude large or very small boxes
#         if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
#             area = cv2.contourArea(contour)
#             if area >230 and area <= 450 and x< 485 and y >250 and y< 370:
#                 pa.append(contour)
#                 cv2.rectangle(image_np,(x+w,y),(x+w+50,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 # checkbox_contours.append(contour)
#             if area >230 and area <= 450 and x> 485 and x<897 and y>250 and y<370:
#                 ja.append(contour)
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#             if area>230 and area<=450 and x>900 and y>250 and y<370:
#                 ja_2.append(contour)
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#             # if area>230 and area<=450 and y>422:
#             #     cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#             #     cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#             #     details.append(contour)
            
#             if area > 230 and area<=450 and x >429 and x<688 and y >496 and y<562:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_pa.append(contour)
            
#             if area > 230 and area<=450 and x >703 and x<999 and y >496 and y<562:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_ja1.append(contour)

#             if area > 230 and area<=450 and x > 1025 and x< 1315 and y >496 and y<562:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_ja2.append(contour)
            
#             if area > 230 and area<=450 and x >429 and x<688 and y >565 and y<626:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_fm_pa.append(contour)

#             if area > 230 and area<=450 and x >703 and x<999 and y >565 and y<626:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_fm_ja1.append(contour)

#             if area > 230 and area<=450 and x >1025 and x<1315 and y >565 and y<626:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_fm_ja2.append(contour)
            
#             if area > 230 and area<=450 and x >429 and x<688 and y >725 and y<880:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_ca_pa.append(contour)

#             if area > 230 and area<=450 and x >703 and x<999 and y >725 and y<880:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_ca_ja1.append(contour)

#             if area > 230 and area<=450 and x >1025 and x<1315 and y >725 and y<880:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
#                 pep_ca_ja2.append(contour)

# for i,c in enumerate(contours):
#     # print(i)
#     x,y,w,h=cv2.boundingRect(c)
#     area=cv2.contourArea(c)
#     if area>5000 and area<10000 and x > 406 and x< 670 and y >590 and y<690:
#         fm_pa_code.append(c)
#         # print(recognize_text_trocr(c))
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#     if area>5000 and area<10000 and x > 680 and x< 980 and y >590 and y<690:
#         fm_ja1_code.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#     if area>5000 and area<10000 and x > 1000 and x< 1380 and y >590 and y<690:
#         fm_ja2_code.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    
#     if area>5000 and area<10000 and x > 406 and x< 670 and y >750 and y<870:
#         ca_pa_code.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#     if area>5000 and area<10000 and x > 680 and x< 980 and y >750 and y<870:
#         ca_ja1_code.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#     if area>5000 and area<10000 and x > 1000 and x< 1380 and y >750 and y<870:
#         ca_ja2_code.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
    
#     if area>5000 and area<10000 and x > 406 and x< 670 and y >880 and y<970:
#         others_pa.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#     if area>5000 and area<10000 and x > 680 and x< 980 and y >880 and y< 970:
#         others_ja1.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

#     if area>5000 and area<10000 and x > 1000 and x< 1380 and y >880 and y<970:
#         others_ja2.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

# text['Principal Applicant'] = yes_no(pa,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner'] = yes_no(ja,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2'] = yes_no(ja_2,img)
# text['Principal Applicant is PEP'] = pep(pep_pa,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner is PEP'] = pep(pep_ja1,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2 is PEP'] = pep(pep_ja2,img)
# text['Principal Applicant is a family member of the PEP'] = pep(pep_fm_pa,img)
# text['Code of pa'] = code(fm_pa_code,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner  is a family member of the PEP'] = pep(pep_fm_ja1,img)
# text['Code of ja1'] = code(fm_ja1_code,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2 is a family member of the PEP'] = pep(pep_fm_ja2,img)
# text['Code of ja2'] = code(fm_ja2_code,img)
# text['Principal Applicant is a close associate of the PEP'] = pep(pep_ca_pa,img)
# text['Code of pa as close associate'] = code(ca_pa_code,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner  is a close associate of the PEP'] = pep(pep_ca_ja1,img)
# text['Code of ja as close associate'] = code(ca_ja1_code,img)
# text['Joint Applicant/ Supplementary Card/ Gurantor/ Registered Owner 2 is a close associate of the PEP'] = pep(pep_ca_ja2,img)
# text['Code of ja2 as close assocaite'] = code(ca_ja2_code,img)
# text['If the relationship between Personal Applicant and PEP is others mention'] = code(others_pa,img)
# text['If the relationship between Joint Applicant and PEP is others mention'] = code(others_ja1,img)
# text['If the relationship between Joint Applicant 2 and PEP is others mention'] = code(others_ja2,img)

# print(text)  
# df = pd.DataFrame(list(text.items()),columns = ['Field','Extracted Text' ])
# print(df)
# plt.imshow(image_np,cmap = 'gray')
# plt.show()

