from curve import curve_detection
import cv2
import numpy as np

cur = curve_detection()
cap = cv2.VideoCapture("project_video.mp4")
while True:
    try:
        ret,frame =cap.read()
        roi_image, mask = cur.roi_mask(frame,300,300)
        hist_x,hist_y = cur.get_histogram(mask)
        histogram = cur.get_curve(hist_y,hist_x,4)
        lef_cur_ran = cur.find_range_curve(histogram,1)
        rt_cur_ran = cur.find_range_curve(histogram,2)
        lef_pts_x,lef_pts_y = cur.find_pts_for_curve_fit(lef_cur_ran,mask,15)
        rt_pts_x, rt_pts_y = cur.find_pts_for_curve_fit(rt_cur_ran,mask,15)

        left_poly = cur.get_curve(lef_pts_x,lef_pts_y,2)
        rt_poly = cur.get_curve(rt_pts_x,rt_pts_y,2)
        roi_image = cur.draw_curve(left_poly,np.arange(0,299,1),roi_image,0)
        roi_image = cur.draw_curve(rt_poly, np.arange(0, 299, 1), roi_image, 0)
        # roi_image = cur.draw_curve(histogram,np.arange(0,299,1),roi_image,2)
        cv2.imshow('roi',roi_image)
        if cv2.waitKey(5) & 0xff == ord('q'):
            break
    except Exception as ex:
        print('second')
        print(ex)

cap.release()
cv2.destroyAllWindows()
