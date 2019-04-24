import cv2
import numpy as np
import sys

colors = np.array([(90,152,44), (129,43,90), (56,68,189)])
positions = [[[225, 375]],
             [[225, 272], [225, 478]],
             [[225, 167], [225, 375], [225, 583]]]
patterns = [cv2.imread('graphics/patterns/empty.png'),
            cv2.imread('graphics/patterns/solid.png'),
            cv2.imread('graphics/patterns/stripe.png')]
outlines = [cv2.imread('graphics/outlines/diamond.png'),
            cv2.imread('graphics/outlines/pill.png'),
            cv2.imread('graphics/outlines/squiggle.png')]
centers = [cv2.imread('graphics/centers/diamond.png'),
            cv2.imread('graphics/centers/pill.png'),
            cv2.imread('graphics/centers/squiggle.png')]

def get_symbol(color, pattern, shape) :
    img = cv2.bitwise_or(outlines[shape], cv2.bitwise_and(centers[shape], patterns[pattern]))
    img[np.where((img!=[0,0,0]).all(axis=2))] = colors[color]
    img[np.where((img==[0,0,0]).all(axis=2))] = (255,255,255)
    return img

def put_symbol(img, symbol, center) :
    tl = center - np.array(symbol.shape[:2]) // 2
    img[tl[0]:tl[0]+symbol.shape[0], tl[1]:tl[1]+symbol.shape[1]] = symbol

def draw_card(color, number, pattern, shape) :
    img = np.zeros((450,750,3), np.uint8)
    img[:,:] = (255,255,255)
    s = get_symbol(color,pattern,shape)
    for p in positions[number] :
        put_symbol(img, s, p)
    return img

def render_scene(cards, contours, shape) :
    img = np.zeros(shape, dtype=np.uint8)
    src = np.array([[0,0],
           [750,0],
           [750,450],
           [0,450]], dtype=np.float32)
    for card, cont in zip(cards, contours) :
        M = cv2.getPerspectiveTransform(src, cont.astype(np.float32))
        raw_card = draw_card(card[0], card[1], card[2], card[3])
        skew_card = cv2.warpPerspective(raw_card, M, (shape[1], shape[0]))
        img = cv2.bitwise_or(img, skew_card)
    return img
