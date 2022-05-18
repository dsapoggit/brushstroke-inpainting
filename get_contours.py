import numpy as np
import cv2 
import random
from scipy.spatial import Delaunay
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt


def get_contours(img_name):
    im = cv2.imread(img_name)
    
    im = cv2.bilateralFilter(im,18, 75, 75)
    
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)

    contours1, hier_1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours2, hier_2 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #print(contours1)
    
    cv2.drawContours(im,contours2,-1,(0,0,255),1)
    
    cv2.imshow('image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   
  
def check_validity(dot, contour):
    temp_hull = np.squeeze(contour, axis = 1) 
    
    if not isinstance(temp_hull, Delaunay):
        temp_hull = Delaunay(temp_hull)

    return temp_hull.find_simplex(np.array(dot)) >= 0

  
def fit_ellipses(contours, dot, drawing):
    good_params = []
 
    for c in contours:
        if  c.shape[0] > 5 and check_validity(dot, c):
            minEllipse = cv2.fitEllipse(c)
            
            if (drawing is not None):
                color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
                cv2.ellipse(drawing, minEllipse, color, 2)
            good_params.append(minEllipse[1]) 
    
    if (drawing is not None):
        cv2.imshow('Contours', drawing)
        
            
    return good_params


def thresh_callback(dot_coord, src_gray, val, need_to_draw = False): 
    threshold = val
    ret, canny_output = cv2.threshold(src_gray, 127, 255, 0)
    
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    drawing = None
    if need_to_draw:
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

    
    good_params = fit_ellipses(contours, dot_coord, drawing)

    return good_params
     
    
def get_params(img_name, n_samples = 1, need_graphics = False, image_gap = 30, color_gap = 25):
    width = []
    length = []
    
    im = cv2.imread(img_name)
    im = cv2.bilateralFilter(im, 15, 75, 75)
    
    H, W, _ = im.shape
    
    good_ellipses = []
    
    for i in range(n_samples):
        #sample dot
        
        x = np.random.randint(image_gap, H - image_gap)
        y = np.random.randint(image_gap, W - image_gap)
        
        dot_coord = [image_gap, image_gap]
        
        dot_col  = im[x, y, :]
        #print(x, y)
        
    
        #get_colour_hsv
        
        hsv = cv2.cvtColor(np.array([[dot_col]]), cv2.COLOR_BGR2HSV)
        
        lower = np.array([max(0, dot_col[0] - color_gap),
                          max(0, dot_col[1] - color_gap), 
                          max(0, dot_col[2] - color_gap)])
                       
        upper = np.array([min(255, dot_col[0] + color_gap),
                          min(255, dot_col[1] + color_gap), 
                          min(255, dot_col[2] + color_gap)])
       
        
        cut_image = cv2.bilateralFilter(im[x - image_gap : x + image_gap, 
                                           y - image_gap : y + image_gap,
                                           :],
                                        15,75,75)
        
        blur = np.uint8(cv2.medianBlur(hsv, 11))
        
        #get countour for this brushstroke and fit ellipse
        
        mask = cv2.inRange(cut_image, lower, upper)
 
        res = cv2.bitwise_and(cut_image, cut_image, mask= mask) 

        if need_graphics :
            cv2.imshow("mask ",cut_image)

        max_thresh = 255
        thresh = 100


        good_ellipses.extend(thresh_callback(dot_coord, mask, thresh, need_graphics))
        
        if need_graphics :
            cv2.waitKey()

    return good_ellipses


def get_ellipse_stats(image_name, n_iters = 50, need_graphics = False):
    good_ellipses = np.array(get_params('./images/stroke_part.jpg', n_iters))
    
    a_mins = []
    a_maxs = []
    
    for el in good_ellipses:
        a_mins.append(min(el))
        a_maxs.append(max(el))
    
    a_mins = np.array(a_mins)
    a_maxs = np.array(a_maxs)
    
    return a_mins.mean(), a_maxs.mean()



def check_image_gap(img_name, n_samples = 10000, need_graphics = False):
    gaps = [5,10, 20, 30, 50, 100]
    mins = []
    maxs = []
    for i in gaps: 
        print(i)
        good_ellipses = get_params(img_name, n_samples, need_graphics, image_gap = i)
    
        a_mins = []
        a_maxs = []
    
        for el in good_ellipses:
            a_mins.append(min(el))
            a_maxs.append(max(el))
    
        mins.append(np.array(a_mins).mean())
        maxs.append(np.array(a_maxs).mean())
        
        print(i)
    
    
    plt.title('Зависимость определения величины мазка от параметров окна')
    plt.ylabel('Величина параметра мазка (меньшая ось)')
    print(mins, [2 * x for x in gaps])
    new_gaps = [2 * x for x in gaps]
    plt.xlabel('Ширина окна')
    plt.plot(mins, new_gaps)


    plt.show()
    plt.clf()
    plt.close()
    
    plt.title('Зависимость определения величины мазка от параметров окна')
    plt.ylabel('Величина параметра мазка (большая ось)')
    plt.xlabel('Ширина окна')
    plt.plot(maxs, new_gaps)


    plt.show()
    plt.clf()
    plt.close()
    
    return 
        
        
def check_color_gap(img_name, n_samples = 10000, need_graphics = False):
    gaps = [5,10, 25, 40, 60, 100]
    mins = []
    maxs = []
    for i in gaps: 
        print(i)
        get_params(img_name, n_samples, need_graphics, image_gap = 30, color_gap = i)
        good_ellipses = get_params(img_name, n_samples, need_graphics, image_gap = i)
    
        a_mins = []
        a_maxs = []
    
        for el in good_ellipses:
            a_mins.append(min(el))
            a_maxs.append(max(el))
    
        mins.append(np.array(a_mins).mean())
        maxs.append(np.array(a_maxs).mean())
    
    
    plt.title('Зависимость определения величины мазка от параметров окна')
    plt.ylabel('Величина параметра мазка (меньшая ось)')
    plt.xlabel('Ширина окна цвета')
    plt.plot(mins, gaps)


    plt.show()
    plt.clf()
    plt.close()
    
    plt.title('Зависимость определения величины мазка от параметров окна')
    plt.ylabel('Величина параметра мазка (большая ось)')
    plt.xlabel('Ширина окна цвета')
    plt.plot(maxs, gaps)


    plt.show()
    plt.clf()
    plt.close()
    
    return     

    
    
        
    
if __name__ == "__main__":
    #print(get_params('./images/Paul-signac-castellane.jpg', 40, need_graphics = True))
    check_color_gap('./images/stroke_part.jpg')
    

    iters = [5, 50, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000]
    #iters = [1, 5, 10]
    anses_i = []
    anses_a = []
    for i in iters:
        mi, ma = get_ellipse_stats('./images/Paul-signac-castellane.jpg', n_iters = i, need_graphics = False)
        anses_i.append(mi)
        anses_a.append(ma)
    
    print(anses_i)
    print(iters)
    plt.title('Меньшая ось эллипса')
    plt.ylabel('Величина параметра мазка')
    plt.xlabel('Итерация')
    plt.plot(iters, anses_i)


    plt.show()
    plt.clf()
    plt.close()
    
    plt.title('Большая ось эллипса')
    plt.ylabel('Величина параметра мазка')
    plt.xlabel('Итерация')
    plt.plot(iters, anses_a)


    plt.show()
    plt.clf()
    plt.close()
    
    print(get_ellipse_stats('./images/stroke_part.jpg', n_iters = 5, need_graphics = False))
    