import cv2
import numpy as np


# Process one frame (same pipeline as image code)
def process_frame(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 80, 150)
    masked = region_of_interest(edges, width, height)
    lines = detect_lines(masked)
    result = fill_lane_region(frame, lines)
    result = cv2.resize(result, (960, 540))
    return result

# Region of Interest function needs width and height for the trapezoid vertices


def region_of_interest(edges, width, height):
    mask = np.zeros_like(edges)
    trapezoid = np.array([[
        (int(width * 0.0),  height),              # bottom-left
        (int(width * 1.0),  height),              # bottom-right
        (int(width * 0.6),  int(height * 0.65)),  # top-right
        (int(width * 0.35), int(height * 0.65))   # top-left
    ]], dtype=np.int32)
    cv2.fillPoly(mask, trapezoid, 255)
    return cv2.bitwise_and(edges, mask)


# Average and Extrapolate lines to get single left and right lane lines
def average_lines(image, lines):
    left_lines, right_lines = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < -0.3:
            left_lines.append((slope, intercept))
        elif slope > 0.3:
            right_lines.append((slope, intercept))
    left_avg = np.mean(left_lines,  axis=0) if left_lines else None
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


# Fill the lane region between the left and right lane lines
def fill_lane_region(image, lines):
    left_avg, right_avg = average_lines(image, lines)
    left_points = make_line_points(image, left_avg)
    right_points = make_line_points(image, right_avg)
    if left_points is None or right_points is None:
        return image
    pts = np.array([
        left_points[0], left_points[1],
        right_points[1], right_points[0]
    ], dtype=np.int32)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.line(result, left_points[0], left_points[1],  (0, 255, 0), 3)
    cv2.line(result, right_points[0], right_points[1], (0, 255, 0), 3)
    return result


# Hough Transform to detect lines in the masked edge image
def detect_lines(masked_edges):
    return cv2.HoughLinesP(
        masked_edges,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=40,
        maxLineGap=150
    )


# Main video processing loop
cap = cv2.VideoCapture('test_video.mp4')

# get video properties for output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = 960
frame_height = 540

# create video writer to save output video
out = cv2.VideoWriter('output_video.mp4', fourcc, fps,
                      (frame_width, frame_height))


if not cap.isOpened():
    print("Error: Could not open video!")
else:
    print("Video opened successfully!")

    while cap.isOpened():
        ret, frame = cap.read()

        # ret = False means video has ended
        if not ret:
            print("Video finished!")
            break

        # run the full pipeline on this frame
        result = process_frame(frame)

        out.write(result)  # save each processed frame to output video

        # show the result
        cv2.imshow('Lane Detection', result)

        # press Q to quit anytime
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
