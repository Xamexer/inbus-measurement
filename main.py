import math
from os import path
import imutils
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
import random as rng


COIN_SIZE = 24.25 #mm

REAL_INBUS_SHORT = 32.5 # mm
REAL_INBUS_LONG = 85 #mm

WINDOW_SIZE = 300
absolute_path = path.dirname(__file__)

final_diff = []

def main(image_nmr):
    relative_path = f"./images/inbus_{image_nmr}.jpg"
    original_image = cv2.imread(path.join(absolute_path, relative_path))
    original_image = imutils.resize(original_image, width=1500)
    height, width, channels = original_image.shape
    
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Original", WINDOW_SIZE, int((height/width)*WINDOW_SIZE))
    cv2.moveWindow("Original", 20 + WINDOW_SIZE * 0,20)
    cv2.imshow("Original", original_image)
    
    blur_amount = 9
    blurred_image = blur_image(original_image, blur_amount)
    
    cv2.namedWindow("Blurred", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Blurred", WINDOW_SIZE, int((height/width)*WINDOW_SIZE))
    cv2.moveWindow("Blurred", 20 + WINDOW_SIZE * 1,20)
    cv2.imshow("Blurred", blurred_image)
    
    threshold = 80
    threshold_image = apply_threshold(blurred_image, threshold)
    kernel = np.ones((5, 5), np.uint8)
    ite = 2
    threshold_image = cv2.erode(threshold_image, kernel, iterations=ite)
    threshold_image = cv2.dilate(threshold_image, kernel, iterations=ite)
    cv2.namedWindow("Binary", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Binary", WINDOW_SIZE, int((height/width)*WINDOW_SIZE))
    cv2.moveWindow("Binary", 20 + WINDOW_SIZE * 2,20)
    cv2.imshow("Binary", threshold_image)
    
    cnts, boxes = get_contours_and_boxes(threshold_image)

    (x1,y1,w1,h1), (x2,y2,w2,h2) = boxes
    pad = 0

    x1a, y1a = max(0, x1-pad), max(0, y1-pad)
    x2a, y2a = max(0, x2-pad), max(0, y2-pad)

    inbus_blurred = blurred_image[y1a:y1+h1+pad,
                                x1a:x1+w1+pad].copy()
    coin_blurred  = blurred_image[y2a:y2+h2+pad,
                                x2a:x2+w2+pad].copy()

    inbus_bgr = cv2.cvtColor(inbus_blurred, cv2.COLOR_GRAY2BGR)
    coin_bgr  = cv2.cvtColor(coin_blurred,  cv2.COLOR_GRAY2BGR)

    cnt0_shift = cnts[0] - [[x1a, y1a]]
    cnt1_shift = cnts[1] - [[x2a, y2a]]

    cv2.drawContours(inbus_bgr, [cnt0_shift], -1, (0,255,0), 1)
    cv2.drawContours(coin_bgr,  [cnt1_shift], -1, (0,255,0), 1)

    cutout_size = 200

    inbus_h, inbus_w, _ = inbus_bgr.shape
    coin_h, coin_w, _ = coin_bgr.shape
    
    cv2.namedWindow("Inbus blurred", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Inbus blurred", cutout_size, int(cutout_size* (inbus_h/inbus_w)))
    cv2.moveWindow("Inbus blurred", 20 + WINDOW_SIZE*2,20 + int((height/width)*WINDOW_SIZE) * 1)
    cv2.imshow("Inbus blurred", inbus_bgr)
    
    cv2.namedWindow("Coin blurred", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Coin blurred", cutout_size, int(cutout_size* (coin_h/coin_w)))
    cv2.moveWindow("Coin blurred", 20,20 + int((height/width)*WINDOW_SIZE) * 1)
    cv2.imshow("Coin blurred", coin_bgr)

    canny_threshold = 100
    edges = cv2.Canny(blurred_image, canny_threshold, canny_threshold * 2)
    edge_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    hough_threshold = 90
    min_line_length = 60
    max_line_gap = 20
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, None, min_line_length, max_line_gap)
    max_length = 0
    for i in range(len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length > max_length:
                max_length = length
                longest_line_angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) + 90
                
    """ print(f"angle longest: {longest_line_angle}") """
    
    angle_difference = 5
    
    longer_sides = []
    longer_side_angles = []
    
    shorter_sides = []
    shorter_side_angles = []
    
    for i in range(len(linesP)):
            x1, y1, x2, y2 = linesP[i][0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) + 90
            if (angle < longest_line_angle+angle_difference and angle > longest_line_angle-angle_difference):
                longer_sides.append(linesP[i][0])
                longer_side_angles.append(angle)
                cv2.line(edge_image, (x1, y1), (x2, y2), (0,0,255), 10, cv2.LINE_AA)
            else:
                shorter_sides.append(linesP[i][0])
                shorter_side_angles.append(angle)
                cv2.line(edge_image, (x1, y1), (x2, y2), (0,255,0), 10, cv2.LINE_AA)
                
    general_long_angle = sum(longer_side_angles)/len(longer_side_angles)
    general_short_angle = sum(shorter_side_angles)/len(shorter_side_angles)
    
    
    #print(f"long angle: {general_long_angle} , short angle: {general_short_angle}")
    cv2.namedWindow("Hough", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hough", WINDOW_SIZE, int((height/width)*WINDOW_SIZE))
    cv2.moveWindow("Hough", 20 + WINDOW_SIZE * 3,20)
    cv2.imshow("Hough", edge_image)


    theta_long  = math.radians(general_long_angle  - 90)
    theta_short = math.radians(general_short_angle - 90)

    u_long  = np.array([math.cos(theta_long),  math.sin(theta_long)],  dtype=np.float32)
    u_short = np.array([math.cos(theta_short), math.sin(theta_short)], dtype=np.float32)

    U = np.stack([u_long, u_short], axis=1)

    M = np.linalg.inv(U)

    H = np.eye(3, dtype=np.float32)
    H[:2, :2] = M

    h_img, w_img = threshold_image.shape[:2]
    corners = np.array([[0,0], [w_img,0], [0,h_img], [w_img,h_img]], dtype=np.float32)
    corners = corners.reshape(-1,1,2)
    corners_T = cv2.perspectiveTransform(corners, H).reshape(-1,2)

    x_min, y_min = corners_T.min(axis=0)
    x_max, y_max = corners_T.max(axis=0)

    out_w = int(np.ceil(x_max - x_min))
    out_h = int(np.ceil(y_max - y_min))


    T = np.eye(3, dtype=np.float32)
    T[0,2] = -x_min
    T[1,2] = -y_min
    H_total = T @ H
    

    cnts, boxes = get_contours_and_boxes(threshold_image)
    (x1,y1,w1,h1), (x2,y2,w2,h2) = boxes
    x2a, y2a = max(0, x2), max(0, y2)
    coin_warped  = threshold_image[y2a:y2+h2,
                                x2a:x2+w2].copy()
    coin_h_px,  coin_w_px  = coin_warped.shape[:2]
    """ scale_y = coin_w_px / coin_h_px
    if abs(scale_y - 1.0) > 0.01:
        S = np.eye(3, dtype=np.float32)
        S[1,1] = scale_y             # Y-Achse strecken
        H_total = S @ H_total        # Streckung VOR die Homographie
        out_h   = int(np.ceil(out_h * scale_y)) """
    
    warped_image = cv2.warpPerspective(threshold_image,
                                   H_total,
                                   (out_w, out_h),
                                   flags=cv2.INTER_LINEAR)

    warped_heigth,warped_width = warped_image.shape

    cv2.namedWindow("Rectified", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rectified", WINDOW_SIZE, int((warped_heigth/warped_width)*WINDOW_SIZE))
    cv2.moveWindow("Rectified", 20 + WINDOW_SIZE * 4,20)
    cv2.imshow("Rectified", warped_image)



    cnts, boxes = get_contours_and_boxes(warped_image)
    (x1,y1,w1,h1), (x2,y2,w2,h2) = boxes
    
    x1a, y1a = max(0, x1), max(0, y1)
    x2a, y2a = max(0, x2), max(0, y2)
    
    
    inbus_warped = warped_image[y1a:y1+h1,
                                x1a:x1+w1].copy()
    coin_warped  = warped_image[y2a:y2+h2,
                                x2a:x2+w2].copy()


    inbus_h_px, inbus_w_px = inbus_warped.shape[:2]
    coin_h_px,  coin_w_px  = coin_warped.shape[:2]

    mm_per_px_x = COIN_SIZE / coin_w_px
    mm_per_px_y = COIN_SIZE / coin_h_px
    #print(mm_per_px_x ,mm_per_px_y)

    proj_x = np.ptp(np.nonzero(inbus_warped)[1])
    proj_y = np.ptp(np.nonzero(inbus_warped)[0])

    long_is_x = proj_x >= proj_y

    if long_is_x:
        inbus_long_px, inbus_short_px = inbus_w_px, inbus_h_px
        mm_per_long,  mm_per_short   = mm_per_px_x, mm_per_px_y
    else:
        inbus_long_px, inbus_short_px = inbus_h_px, inbus_w_px
        mm_per_long,  mm_per_short   = mm_per_px_y, mm_per_px_x

    inbus_long_mm  = inbus_long_px  * mm_per_long
    inbus_short_mm = inbus_short_px * mm_per_short

    diff_long  = REAL_INBUS_LONG  - inbus_long_mm
    diff_short = REAL_INBUS_SHORT - inbus_short_mm

    real_total     = REAL_INBUS_LONG + REAL_INBUS_SHORT
    measured_total = inbus_long_mm   + inbus_short_mm


    diff_mm_dir   = measured_total - real_total
    diff_pct_abs  = abs(diff_mm_dir) / real_total * 100

    abs_diff_mm   = abs(diff_long) + abs(diff_short)
    abs_diff_pct  = abs_diff_mm / real_total * 100


    warped_vis = cv2.cvtColor(warped_image, cv2.COLOR_GRAY2BGR)

    inbus_pt1 = (x1a,          y1a)
    inbus_pt2 = (x1a + w1,     y1a + h1)

    coin_pt1  = (x2a,          y2a)
    coin_pt2  = (x2a + w2,     y2a + h2)

    cv2.rectangle(warped_vis, inbus_pt1, inbus_pt2, (0, 0, 255), 8)
    cv2.rectangle(warped_vis, coin_pt1,  coin_pt2,  (0, 255, 0), 8)

    cv2.namedWindow("Warped + Boxes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Warped + Boxes", WINDOW_SIZE*2,
                    int((warped_heigth/warped_width)*WINDOW_SIZE*2))
    cv2.moveWindow("Warped + Boxes", 20+WINDOW_SIZE*4,20 + int((height/width)*WINDOW_SIZE) * 1)
    cv2.imshow("Warped + Boxes", warped_vis)
    
    long_term = "perfect"
    short_term = "perfect"
    
    if diff_long > 0:
        long_term = "kurz"
    else:
        long_term = "lang"
        
    if diff_short > 0:
        short_term = "kurz"
    else:
        short_term = "lang"
        
    # 5) Ausgabe
    print(f"\n--- Zusammenfassung {image_nmr} ---")
    print(f"Langer Schenkel : {inbus_long_mm :.2f} mm"
        f"  (Δ = {diff_long :+6.2f} mm, also zu {long_term})")
    print(f"Kurzer Schenkel : {inbus_short_mm:.2f} mm"
        f"  (Δ = {diff_short:+6.2f} mm, also zu {short_term})")
    #print(f"Gesamt Δ        : {diff_mm_dir:+6.2f} mm  "
    #    f"({diff_pct_abs:5.2f} %)  – gerichteter Fehler")
    print(f"Gesamt |Δ|      : {abs_diff_mm:6.2f} mm  "
        f"({abs_diff_pct:5.2f} %)  - Summe der Beträge")


    final_diff.append(abs_diff_pct)


    H_inv = np.linalg.inv(H_total)

    def warp_box_to_original(pt1, w, h, Hinv):
        """pt1 = (x, y) links-oben im Warp-Bild; w/h = Breite/Höhe"""
        pts = np.array([
            [pt1[0],       pt1[1]],        # links-oben
            [pt1[0]+w,     pt1[1]],        # rechts-oben
            [pt1[0]+w,     pt1[1]+h],      # rechts-unten
            [pt1[0],       pt1[1]+h]       # links-unten
        ], dtype=np.float32).reshape(-1,1,2)
        return cv2.perspectiveTransform(pts, Hinv).reshape(-1,2).astype(int)

    inbus_poly = warp_box_to_original((x1a, y1a), w1, h1, H_inv)
    coin_poly  = warp_box_to_original((x2a, y2a), w2, h2, H_inv)

    orig_vis = original_image.copy()

    cv2.polylines(orig_vis, [inbus_poly], True, (0,0,255),  2, cv2.LINE_AA)   # ROT
    cv2.polylines(orig_vis, [coin_poly],  True, (0,255,0),  2, cv2.LINE_AA)   # GRÜN


    PAD = 20

    all_pts = np.vstack([inbus_poly, coin_poly])

    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    h_img, w_img = orig_vis.shape[:2]
    x_min = max(int(x_min - PAD), 0)
    y_min = max(int(y_min - PAD), 0)
    x_max = min(int(x_max + PAD), w_img - 1)
    y_max = min(int(y_max + PAD), h_img - 1)

    zoom_vis = orig_vis[y_min:y_max, x_min:x_max].copy()

    cv2.namedWindow("Zoom + Boxes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Zoom + Boxes",
                    WINDOW_SIZE,
                    int((y_max - y_min) / (x_max - x_min) * WINDOW_SIZE))
    cv2.moveWindow("Zoom + Boxes", 20 + WINDOW_SIZE * 5, 20)
    cv2.imshow("Zoom + Boxes", zoom_vis)



    
    cv2.waitKey(0)


def blur_image(image: np.ndarray, amount: int) -> np.ndarray:
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grey_image, (amount, amount), 0)
    return blurred_image

def apply_threshold(image: np.ndarray, threshold: int) -> np.ndarray:
    threshold_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    return threshold_image

def get_contours_and_boxes(bin_img, n=2):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:n]
    boxes = [cv2.boundingRect(c) for c in cnts]
    return cnts, boxes


if __name__ == "__main__":
    for i in range(1,6):
        main(i)
    print("\n\n------------------\n")
    print(f"{sum(final_diff)/len(final_diff):5.2f}% total difference")
    print("\n------------------")
