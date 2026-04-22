import cv2
import numpy as np
import os

# Read images from folder path
folder_path = "./images"
valid_extensions = ('.jpg', '.jpeg', '.png')

# Get all valid image paths
image_paths = [
    os.path.join(folder_path, file)
    for file in sorted(os.listdir(folder_path))
    if file.lower().endswith(valid_extensions)
]

# 2–8 images required
if not (2 <= len(image_paths) <= 8):
    print("2-8 images required in the folder.")
    exit()
print(f"{len(image_paths)} images in folder.")

images = []
for path in image_paths:
    img = cv2.imread(path)
    images.append(img)

# Display the original input images
if len(images) > 1:
    input_display = cv2.hconcat(images)
    cv2.imshow("Original input Images", input_display)

#Initialize the result with the first image
result = images[0]

# Process the other images
for i in range(len(images)-1):
    print(f"Processing image pair {i+1} & {i+2}...")

    # Current pair of images to stitch
    img1 = result 
    img2 = images[i+1]

    # Detect features using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    print(f"Found {len(good_matches)} good matches")

    # Check if at least 20 good matches have been found
    if len(good_matches) < 20:
        print("Need to find more good match")
        exit()

    # Extract matched points
    pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    # Compute homography using RANSAC
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    #Warp the second image to align with the first
    height, width = img1.shape[:2]
    warped_img = cv2.warpPerspective(img2, H, (width, height))

    # Create a panorama canvas
    panorama = cv2.warpPerspective(img2, H, (width * 2, height))
    panorama[0:height, 0:width] = img1

    # Change the panorama with a wider blending region
    mask = np.sum(warped_img, axis=2) > 0
    for y in range(height):
        for x in range(width):
            if x < width and y < height and mask[y, x]:

                # For overlapping regions, use a weighted blend based on position
                if x < width and y < height and np.sum(img1[y, x]) > 0:

                    # The closer to the edge of img1, the more weight for warped_img
                    weight = min(1.0, max(0.0, (x - (width - 10)) / 10)) if width > 10 else 0.5

                    # Apply the standard linear interpolation formula
                    panorama[y, x] = ((1-weight) * img1[y, x].astype(float) + weight * warped_img[y, x].astype(float)).astype(np.uint8)
                else:
                    # Just use the warped image where there's no overlap
                    panorama[y, x] = warped_img[y, x]

    # Update the result
    result = panorama

#Crop the black areas from the panorama 
gray_panorama = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) # Convert to grayscale
_, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY) # Threshold to create a binary mask
coords = cv2.findNonZero(thresh) # Find all non-zero points
x, y, w, h = cv2.boundingRect(coords)  # Find the bounding box
cropped_panorama = result[y:y+h, x:x+w]  # Crop the image

# Display the results
cv2.imshow("Panorama (With Black Areas)", result)
cv2.imshow("Panorama", cropped_panorama)
cv2.imwrite("panorama.jpg", result)
cv2.imwrite("cropped_panorama.jpg", cropped_panorama)

cv2.waitKey(0)
cv2.destroyAllWindows()