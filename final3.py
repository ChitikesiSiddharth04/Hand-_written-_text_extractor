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
# images = convert_from_path("C:\\Users\\gurra\\Desktop\\CIMB\\OCR\\Different handwritings\\File 2 - Copy.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:\\Program Files (x86)\\poppler-24.07.0\\Library\\bin")  # Set dpi for higher resolution
# images = convert_from_path("C:/Users/DELL/Downloads/Rahmat.pdf", dpi=170, first_page=1, last_page=11,poppler_path="C:/Users/DELL/Downloads/poppler-24.07.0/Library/bin")  # Set dpi for higher resolution

# image_np = np.array(images[2])
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
#                 # if distance <10:
#                 #     print(distance, i, j)
                
#                 if distance <= min_distance:
#                     keep = False
#                     to_remove.add(j)
#                     break
        
#         if keep:
#             filtered_contours.append(contour1)

#     # print(to_remove)
#     contours = [c for i, c in enumerate(contours) if i not in to_remove]
#     return contours

#IMAGE FOR FIELDS
def is_ticked(roi):
    roi=cv2.resize(roi,(50,73))
    _, binary = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    black_pixel_ratio = black_pixels / total_pixels
    # print(black_pixel_ratio)
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



def tax_number(contours,img):
    result = []
    for i,c in enumerate(contours):
        # print(i)
        x,y,w,h = cv2.boundingRect(c)
        roi_image = img[y:y+h, x:x+w]
        # t = is_ticked(roi_image)
        # cv2.rectangle(image_np,(x,y+53),(x+260,y+75),(0,255,0),3)
        # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        if is_ticked(roi_image):
        #     print(i)
            if i == 1:
                result.append(recognize_text_trocr(img[y+53:y+75,x:x+260]))
                # cv2.rectangle(image_np,(x,y+53),(x+260,y+75),(0,255,0),3)
                # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            else:
                result.append("Not Applicable ")
    
    return result

# us_resident_pa = []
# us_resident_ja1 = []
# us_resident_ja2 = []
# us_permanent_resident_pa = []
# us_permanent_resident_ja1 = []
# us_permanent_resident_ja2 = []
# us_citizen_pa = []
# us_citizen_ja1 = []
# us_citizen_ja2 = []
# field_contours=[]

# contours_cb = remove_nearby_contours(contours_cb)
# for i,contour in enumerate(contours_cb):
#     x, y, w, h = cv2.boundingRect(contour)
#     aspect_ratio = w / float(h)
#     if 0.8 <= aspect_ratio <= 1.5:  # Aspect ratio close to 1 for squares
#         # Filter based on size to exclude large or very small boxes
#         if 5<= w <=20  and 5<= h <= 20:  # Adjust these size thresholds as needed
#             area = cv2.contourArea(contour)
#             if area >250 and area <= 450 and x>400 and x <666 and y>=1089 and y< 1190:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 # cv2.rectangle(image_np,(x,y+50),(x+260,y+80),(0,255,0),3)
#                 # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_resident_pa.append(contour)
#             elif area >250 and area <= 450 and x > 666 and x<978  and y>= 1089 and y < 1190:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_resident_ja1.append(contour)
#             elif area >250 and area <= 450 and x>1000 and y>=1089 and y< 1190:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_resident_ja2.append(contour)
#             elif area >250 and area <= 450 and x>400 and x <666 and y>=1200 and y< 1300:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_citizen_pa.append(contour)
#             elif area >250 and area <= 450 and x>666 and x <978 and y>=1200 and y< 1300:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_citizen_ja1.append(contour)
#             elif area >250 and area <= 450 and x>1000 and y>=1200 and y< 1300:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 # cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_citizen_ja2.append(contour)
#             elif area >250 and area <= 450 and x>400 and x <666 and y>=1300 and y< 1405:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_permanent_resident_pa.append(contour)
#             elif area >250 and area <= 450 and x>666 and x <978 and y>=1300 and y< 1405:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_permanent_resident_ja1.append(contour)
#             elif area >250 and area <= 450 and x>1000 and x< 1300 and y>=1300 and y< 1405:
#                 cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#                 cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
#                 us_permanent_resident_ja2.append(contour)
            
# text = {}
# f_c = []

# print(len(us_permanent_resident_ja1),len(us_permanent_resident_ja2),len(us_permanent_resident_pa))
# # print(len(us_resident_pa))
# text['tax number if principal applicant is US Resident '] = tax_number(us_resident_pa,img)
# text['tax number if joint applicant 1 is US Resident '] = tax_number(us_resident_ja1,img)
# text['tax number if joint applicant 2 is US Resident '] = tax_number(us_resident_ja2,img)
# text['tax number if principal applicant is US Citizen '] = tax_number(us_citizen_pa,img)
# text['tax number if joint applicant 1 is US Citizen '] = tax_number(us_citizen_ja1,img)
# text['tax number if joint applicant 2 is US Citizen '] = tax_number(us_citizen_ja2,img)
# text['tax number if principal applicant is US Permanent Resident '] = tax_number(us_permanent_resident_pa,img)
# text['tax number if joint applicant 1 is US Resident '] = tax_number(us_permanent_resident_ja1,img)
# text['tax number if joint applicant 2 is US Resident '] = tax_number(us_permanent_resident_ja2,img)

# contours_keep = remove_nearby_contours(contours)
# for i,c in enumerate(contours_keep):
#     x,y,w,h=cv2.boundingRect(c)
#     area=cv2.contourArea(c)
#     if area>15000 and area<30000 and x> 300 and y> 1390:
#         roi_image = img[y:y+h, x:x+w]
#         generated_text = recognize_text_trocr(roi_image)
#         text[f'Field_{i}'] = generated_text
        
#         f_c.append(c)
#         cv2.rectangle(image_np,(x,y),(x+w,y+h),(0,255,0),3)
#         cv2.putText(image_np,str(i),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

# for c in f_c:
#     res,field=field_detection(c,img)
#     if field in text.keys():
#         field=field+"2"
#     text[field]=res

# dec = declaration(checkbox_contours[:4],img)
# # if not dec:
#     # text['declaration'] = 'No'
# # else:
# text['declaration'] = dec

# print(text)

# order_text = dict(sorted(text.items(),reverse = True))

# df = pd.DataFrame(list(order_text.items()),columns = ['Field','Extracted Text' ])
# # df_t = df.transpose()
# print(df)
# # # df.to_csv('thomas5.csv',index = False)

# plt.imshow(image_np,cmap = 'gray')
# plt.show()
