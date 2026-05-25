import cv2
import numpy as np

# Load the image
image = cv2.imread('test_image.jpg')
height, width = image.shape[:2]
print(f"Image size: width={width}, height={height}")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# Canny Edge Detection
edges = cv2.Canny(blur, 80, 150)

# Region of Interest (Trapezoid) - Adjust the vertices based on the image size


def region_of_interest(edges):
    mask = np.zeros_like(edges)
    trapezoid = np.array([[
        (int(width * 0.0),  height),            # bottom-left
        (int(width * 1.0),  height),            # bottom-right
        (int(width * 0.6),  int(height * 0.65)),  # top-right
        (int(width * 0.35), int(height * 0.65))  # top-left
    ]], dtype=np.int32)
    cv2.fillPoly(mask, trapezoid, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges, trapezoid


masked_edges, trapezoid = region_of_interest(edges)


# Average and Extrapolate lines to get single left and right lane lines
def average_lines(image, lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # ignore horizontal and vertical lines
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # negative slope = left lane, positive slope = right lane
        if slope < -0.3:
            left_lines.append((slope, intercept))
        elif slope > 0.3:
            right_lines.append((slope, intercept))

    # average all left line segments into one line
    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None
    return left_avg, right_avg


def make_line_points(image, line_params):
    if line_params is None:
        return None
    slope, intercept = line_params
    height = image.shape[0]
    y1 = height
    y2 = int(y1 * 0.6)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return (x1, y1), (x2, y2)


def draw_lane_lines(image, lines):
    left_avg, right_avg = average_lines(image, lines)
    line_image = image.copy()

    left_points = make_line_points(image, left_avg)
    right_points = make_line_points(image, right_avg)

    if left_points:
        cv2.line(line_image, left_points[0], left_points[1], (0, 255, 0), 8)
    if right_points:
        cv2.line(line_image, right_points[0], right_points[1], (0, 255, 0), 8)

    return line_image


# Fill Lane region with color
def fill_lane_region(image, lines):
    left_avg, right_avg = average_lines(image, lines)

    left_points = make_line_points(image, left_avg)
    right_points = make_line_points(image, right_avg)

    if left_points is None or right_points is None:
        return image

    # 4 corners of the lane region
    pts = np.array([
        left_points[0],  # bottom-left
        left_points[1],  # top-left
        right_points[1],  # top-right
        right_points[0]  # bottom-right
    ], dtype=np.int32)

    # create empty overlay and fill the lane region with color
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))  # green color with full opacity

    # blend the overlay with the original image (0.3 = 30% opacity)
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

    # draw the lane lines on top of the filled region
    cv2.line(result, left_points[0], left_points[1], (0, 255, 0), 2)
    cv2.line(result, right_points[0], right_points[1], (0, 255, 0), 2)
    return result


# Hough Transform to detect lines
def detect_lines(masked_edges):
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=150
    )
    return lines


lines = detect_lines(masked_edges)


# draw lines on the original image
line_image = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    print(f"Total lines detected: {len(lines)}")
else:
    print("No lines detected.")


# DEBUG: Draw trapezoid on image to check position and vertices
debug_image = image.copy()
cv2.polylines(debug_image, trapezoid, isClosed=True,
              color=(0, 255, 0), thickness=3)
for point in trapezoid[0]:
    cv2.circle(debug_image, tuple(point), 10, (0, 0, 255), -1)


lines = detect_lines(masked_edges)
# result = draw_lane_lines(image, lines)
result = fill_lane_region(image, lines)

# Display the results in separate windows
cv2.imshow('1 - Original', image)
# cv2.imshow('2 - Grayscale', gray)
# cv2.imshow('3 - Blurred', blur)
# cv2.imshow('4 - Edges', edges)
# cv2.imshow('5 - Masked Edges', masked_edges)
# cv2.imshow('6 - Trapezoid Position', debug_image)
# cv2.imshow('7 - Detected Lines', line_image)
cv2.imshow('8 - Lane Detection Result', result)

# save the result image
cv2.imwrite('output_image.jpg', result)
print("Output image")
cv2.waitKey(0)
cv2.destroyAllWindows()
