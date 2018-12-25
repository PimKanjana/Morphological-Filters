import cv2
import numpy as np

img = cv2.imread('im.jpg',0)
kernel = np.ones((5,5),np.uint8)

# Erosion
erosion = cv2.erode(img,kernel,iterations = 1)
compare_ero = np.hstack((img,erosion))
cv2.imshow('Erosion',compare_ero)
cv2.imwrite('Erosion.jpg',erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dilation
dilation = cv2.dilate(img,kernel,iterations = 1)
compare_dil = np.hstack((img,dilation))
cv2.imshow('Dilation',compare_dil)
cv2.imwrite('Dilation.jpg',dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
compare_open = np.hstack((img,opening))
cv2.imshow('Opening',compare_open)
cv2.imwrite('Opening.jpg',opening)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Closing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
compare_close = np.hstack((img,closing))
cv2.imshow('Closing',compare_close)
cv2.imwrite('Closing.jpg',closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Erosion vs Dilation vs Opening
compare_ero_dil_open = np.hstack((erosion,dilation,opening))
cv2.imshow('Erosion vs Dilation vs Opening',compare_ero_dil_open)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Dilation vs Erosion vs Closing
compare_dil_ero_close = np.hstack((dilation,erosion,closing))
cv2.imshow('Dilation vs Erosion vs Closing',compare_dil_ero_close)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opening vs Closing
compare_open_close = np.hstack((opening,closing))
cv2.imshow('Opening vs Closing',compare_open_close)
cv2.waitKey(0)
cv2.destroyAllWindows()


## Structuring Element

# Rectangular Kernel
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# Elliptical Kernel
ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Cross-shaped Kernel
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# compare each kernel with closing
rect_op = cv2.morphologyEx(img, cv2.MORPH_CLOSE, rect_kernel)
ellip_op = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellip_kernel)
cross_op = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cross_kernel)
compare_rect_ellip_cross = np.hstack((rect_op,ellip_op,cross_op))
cv2.imshow('Rectangular vs Elliptical vs Cross-shaped Kernel with Closing Morpholoical Filter',compare_rect_ellip_cross)
cv2.waitKey(0)
cv2.destroyAllWindows()
