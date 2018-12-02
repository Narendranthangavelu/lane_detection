import cv2 as cv2
import numpy as np

class curve_detection():
    def __init__ (self):
        #defining hsv range for yellow and white line detection
        self.low_y = np.array([20, 150, 150])
        self.high_y = np.array([30, 255, 255])
        self.low_w = np.array([0, 0, 210])
        self.high_w = np.array([255, 30, 255])
        self.width_roi = 300
        self.height_roi = 300
        self.oldroot=[0,40,130,200]
    def mid_pt(self,poly1 ,poly2,y_pos):
        #to find the mid point between two curves
        #at a specified y position
        x1 = poly1(y_pos)
        x2 = poly2(y_pos)
        m_p = np.int32((x1+x2)/2)
        return [m_p,y_pos]
    def find_range_curve(self, histogram, curve_no=1):
        derivation = histogram.deriv()
        #to ge                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          t the x positon where slope is zero
        roots = np.int32(np.roots(derivation.coefficients))
        # only getting first and last value
        roots_ = [roots[np.argmin(roots)],roots[np.argmax(roots)]]
        y_val =  0.5*histogram(roots_[curve_no-1])
        hist2= histogram-y_val
        root =  (np.roots(hist2.coefficients))
        if root[0].imag > 0 or root[2].imag >0:
            root = self.oldroot
        root = np.sort(root)
        root = np.int32(root)
        self.oldroot = root
        root[np.where(root < 0 )]=0
        root[np.where(root>299)]=0
        if curve_no == 1:
            curve_locaitons = [root[0],root[1]]
        else:
            curve_locaitons = [root[2], root[3]]
        return curve_locaitons
    def find_pts_for_curve_fit(self,ranges,mask,interval=10):
        points_x = []
        points_y = []
        i = 0
        while i < 300:
            loc = np.where(mask[i:i + interval, ranges[0]:ranges[1]])
            if len(loc[0]) > 0:
                points_x.append((np.average(loc[1])) + ranges[0])
                points_y.append(i)
            i += interval
        return (points_x, points_y)
    def roi_mask(self,image,width,height):
        Shape = image.shape
        lowlefx = Shape[1] * 0.10
        lowrigx = Shape[1] * 0.95
        uplefx = Shape[1] * 0.40
        uprigx = Shape[1] * 0.60
        upy = Shape[0] * 0.65
        downy = Shape[0] * 0.95
        pts1 = np.float32([[uplefx, upy], [lowlefx, downy], [uprigx, upy], [lowrigx, downy]])
        pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, (300, 300))
        hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
        mask = self.hsv_inrange(hsv)
        return dst,mask
    def hsv_inrange(self, image_hsv):
        mask1 = cv2.inRange(image_hsv, self.low_w, self.high_w)
        mask2 = cv2.inRange(image_hsv, self.low_y, self.high_y)
        mask = np.bitwise_or(mask1, mask2)
        return mask
    def get_curve(self,x_pts,y_pts,degree=2):
        poly =np.poly1d(np.polyfit(y_pts, x_pts,degree))
        return poly
    def draw_curve(self,poly,pts,image,axis=0):
        curved_image=image
        for c in pts:
            if axis == 0:
                cv2.circle(curved_image, (np.int32(poly(c)),
                            np.int32(c)), 3, (0, 255, 25), -1)
            else:
                cv2.circle(curved_image,(np.int32(c), (np.int32(poly(c))))
                           , 3, (0, 255, 25), -1)
        return curved_image
    def get_histogram(self,mask):
        edges = cv2.Sobel(mask, cv2.CV_8U, 1, 0, ksize=5)
        count_list = np.count_nonzero(edges, axis=0)
        p1 = np.arange(0, 300, 1)
        return p1,count_list




