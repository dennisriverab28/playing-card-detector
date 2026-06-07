# Import necessary packages
import numpy as np
import cv2
import time

# --- OpenCV 3/4 compatibility for findContours ---
def find_contours_compat(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE):
    """
    Returns (contours, hierarchy) for both OpenCV 3 and 4.
    """
    res = cv2.findContours(img, mode, method)
    if len(res) == 3:
        # OpenCV 3.x
        _, cnts, hier = res
    else:
        # OpenCV 4.x
        cnts, hier = res
    return cnts, hier


### Constants ###

# Adaptive threshold levels
BKG_THRESH = 60
CARD_THRESH = 30

# Width and height of card corner, where rank and suit are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

RANK_DIFF_MAX = 2000
SUIT_DIFF_MAX = 700

CARD_MAX_AREA = 120000
CARD_MIN_AREA = 25000

font = cv2.FONT_HERSHEY_SIMPLEX

### Structures to hold query card and train card information ###

class Query_card:
    """Structure to store information about query cards in the camera image."""

    def __init__(self):
        self.contour = []           # Contour of card
        self.width, self.height = 0, 0  # Width and height of card
        self.corner_pts = []        # Corner points of card
        self.center = []            # Center point of card
        self.warp = []              # 200x300, flattened, grayed, blurred image
        self.rank_img = []          # Thresholded, sized image of card's rank
        self.suit_img = []          # Thresholded, sized image of card's suit
        self.best_rank_match = "Unknown"  # Best matched rank
        self.best_suit_match = "Unknown"  # Best matched suit
        self.rank_diff = 0          # Difference vs best matched train rank image
        self.suit_diff = 0          # Difference vs best matched train suit image

class Train_ranks:
    """Structure to store information about train rank images."""

    def __init__(self):
        self.img = []   # Thresholded, sized rank image loaded from disk
        self.name = "Placeholder"

class Train_suits:
    """Structure to store information about train suit images."""

    def __init__(self):
        self.img = []   # Thresholded, sized suit image loaded from disk
        self.name = "Placeholder"

### Functions ###
def load_ranks(filepath):
    """Loads rank images from directory specified by filepath. Stores
    them in a list of Train_ranks objects."""

    train_ranks = []
    i = 0
    
    for Rank in ['Ace','Two','Three','Four','Five','Six','Seven',
                 'Eight','Nine','Ten','Jack','Queen','King']:
        train_ranks.append(Train_ranks())
        train_ranks[i].name = Rank
        filename = Rank + '.jpg'
        train_ranks[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_ranks

def load_suits(filepath):
    """Loads suit images from directory specified by filepath. Stores
    them in a list of Train_suits objects."""

    train_suits = []
    i = 0
    
    for Suit in ['Spades','Diamonds','Clubs','Hearts']:
        train_suits.append(Train_suits())
        train_suits[i].name = Suit
        filename = Suit + '.jpg'
        train_suits[i].img = cv2.imread(filepath+filename, cv2.IMREAD_GRAYSCALE)
        i = i + 1

    return train_suits

def preprocess_image(image):
    """Returns a grayed, blurred, and adaptively thresholded camera image."""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # Adaptive threshold based on a background pixel to handle lighting changes
    img_w, img_h = np.shape(image)[:2]  # note: names follow original repo style
    bkg_level = gray[int(img_h/100)][int(img_w/2)]
    thresh_level = bkg_level + BKG_THRESH

    retval, thresh = cv2.threshold(blur,thresh_level,255,cv2.THRESH_BINARY)
    
    return thresh

def find_cards(thresh_image):
    """Finds all card-sized contours in a thresholded camera image.
    Returns the contours sorted largest->smallest and an array marking which are cards."""

    # Find contours and sort their indices by contour size
    cnts, hier = find_contours_compat(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_sort = sorted(range(len(cnts)), key=lambda i : cv2.contourArea(cnts[i]), reverse=True)

    # If there are no contours, do nothing
    if len(cnts) == 0:
        return [], []
    
    # Otherwise, initialize empty sorted contour and hierarchy lists
    cnts_sort = []
    hier_sort = []
    cnt_is_card = np.zeros(len(cnts), dtype=int)

    # Fill empty lists with sorted contour and sorted hierarchy.
    for i in index_sort:
        cnts_sort.append(cnts[i])
        hier_sort.append(hier[0][i])

    # Determine which of the contours are cards by applying the criteria:
    # 1) area between [CARD_MIN_AREA, CARD_MAX_AREA]
    # 2) no parent in hierarchy (hier_sort[i][3] == -1)
    # 3) approximates to 4 corners
    for i in range(len(cnts_sort)):
        size = cv2.contourArea(cnts_sort[i])
        peri = cv2.arcLength(cnts_sort[i], True)
        approx = cv2.approxPolyDP(cnts_sort[i], 0.01*peri, True)
        
        if ((size < CARD_MAX_AREA) and (size > CARD_MIN_AREA)
            and (hier_sort[i][3] == -1) and (len(approx) == 4)):
            cnt_is_card[i] = 1

    return cnts_sort, cnt_is_card

def preprocess_card(contour, image):
    """Uses contour to find information about the query card. Isolates rank
    and suit images from the card."""

    # Initialize new Query_card object
    qCard = Query_card()

    qCard.contour = contour

    # Find perimeter of card and use it to approximate corner points
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01*peri, True)
    pts = np.float32(approx)
    qCard.corner_pts = pts

    # Find width and height of card's bounding rectangle
    x,y,w,h = cv2.boundingRect(contour)
    qCard.width, qCard.height = w, h

    # Center point of card by averaging corners
    average = np.sum(pts, axis=0)/len(pts)
    cent_x = int(average[0][0])
    cent_y = int(average[0][1])
    qCard.center = [cent_x, cent_y]

    # Warp card into 200x300 flattened grayscale image
    qCard.warp = flattener(image, pts, w, h)

    # Grab corner of warped card image and 4x zoom
    Qcorner = qCard.warp[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    # Sample white pixel intensity to set threshold level
    white_level = Qcorner_zoom[15, int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESH
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2.THRESH_BINARY_INV)
    
    # Split into top (rank) and bottom (suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    # ---- Rank contours (OpenCV 4 compat) ----
    Qrank_cnts, hier = find_contours_compat(Qrank, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea, reverse=True)

    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH, RANK_HEIGHT), 0, 0)
        qCard.rank_img = Qrank_sized

    # ---- Suit contours (OpenCV 4 compat) ----
    Qsuit_cnts, hier = find_contours_compat(Qsuit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea, reverse=True)
    
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        qCard.suit_img = Qsuit_sized

    return qCard

def match_card(qCard, train_ranks, train_suits):
    """Finds best rank and suit matches for the query card by differencing
    the query images against the training images and picking the smallest sum."""

    best_rank_match_diff = 10000
    best_suit_match_diff = 10000
    best_rank_match_name = "Unknown"
    best_suit_match_name = "Unknown"

    # Skip if no symbol images (e.g., detection failed)
    if (len(qCard.rank_img) != 0) and (len(qCard.suit_img) != 0):
        # Rank
        for Trank in train_ranks:
            diff_img = cv2.absdiff(qCard.rank_img, Trank.img)
            rank_diff = int(np.sum(diff_img)/255)
            if rank_diff < best_rank_match_diff:
                best_rank_diff_img = diff_img
                best_rank_match_diff = rank_diff
                best_rank_name = Trank.name

        # Suit
        for Tsuit in train_suits:
            diff_img = cv2.absdiff(qCard.suit_img, Tsuit.img)
            suit_diff = int(np.sum(diff_img)/255)
            if suit_diff < best_suit_match_diff:
                best_suit_diff_img = diff_img
                best_suit_match_diff = suit_diff
                best_suit_name = Tsuit.name

    # Threshold to accept best matches
    if (best_rank_match_diff < RANK_DIFF_MAX):
        best_rank_match_name = best_rank_name

    if (best_suit_match_diff < SUIT_DIFF_MAX):
        best_suit_match_name = best_suit_name

    # Return the identity and the match quality numbers
    return best_rank_match_name, best_suit_match_name, best_rank_match_diff, best_suit_match_diff

def draw_results(image, qCard):
    """Draw the card name, center point, and contour on the camera image."""

    x = qCard.center[0]
    y = qCard.center[1]
    cv2.circle(image, (x,y), 5, (255,0,0), -1)

    rank_name = qCard.best_rank_match
    suit_name = qCard.best_suit_match

    # Draw card name twice, so letters have black outline
    cv2.putText(image, (rank_name+' of'), (x-60,y-10), font, 1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(image, (rank_name+' of'), (x-60,y-10), font, 1, (50,200,200), 2, cv2.LINE_AA)

    cv2.putText(image, suit_name, (x-60,y+25), font, 1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(image, suit_name, (x-60,y+25), font, 1, (50,200,200), 2, cv2.LINE_AA)

    # Debug diffs (optional)
    # r_diff = str(qCard.rank_diff); s_diff = str(qCard.suit_diff)
    # cv2.putText(image, r_diff, (x+20,y+30), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
    # cv2.putText(image, s_diff, (x+20,y+50), font, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return image

def flattener(image, pts, w, h):
    """Flattens an image of a card into a top-down 200x300 perspective."""
    temp_rect = np.zeros((4,2), dtype="float32")
    
    s = np.sum(pts, axis=2)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=-1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    # Order points as [TL, TR, BR, BL] depending on orientation
    if w <= 0.8*h:  # vertical
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.2*h:  # horizontal
        temp_rect[0] = bl
        temp_rect[1] = tl
        temp_rect[2] = tr
        temp_rect[3] = br

    # diamond orientation
    if w > 0.8*h and w < 1.2*h:
        if pts[1][0][1] <= pts[3][0][1]:  # tilt left
            temp_rect[0] = pts[1][0]  # TL
            temp_rect[1] = pts[0][0]  # TR
            temp_rect[2] = pts[3][0]  # BR
            temp_rect[3] = pts[2][0]  # BL
        else:  # tilt right
            temp_rect[0] = pts[0][0]  # TL
            temp_rect[1] = pts[3][0]  # TR
            temp_rect[2] = pts[2][0]  # BR
            temp_rect[3] = pts[1][0]  # BL
            
    maxWidth  = 200
    maxHeight = 300

    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], np.float32)
    M = cv2.getPerspectiveTransform(temp_rect, dst)
    warp = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    return warp
