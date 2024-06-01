import cv2
import os
import pytesseract
import matplotlib.pyplot as plt
import easyocr


# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=True)

path = "F:/FYP_Programe/test2/save/"

img = cv2.imread(path+'MDU.jpg')

# Perform OCR on the thresholded image
license_plate_text = reader.readtext(img)

plt.imshow(img)
plt.show()

print(license_plate_text)
