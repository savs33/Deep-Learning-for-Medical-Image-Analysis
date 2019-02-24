import shutil
import os
import numpy as np
np.random.seed(42)
import cv2
from tqdm import tqdm


def copy_files(srcs,dsts):
    for src,dst in tqdm(zip(srcs,dsts),total=len(srcs)):
        shutil.copy(src,dst)

def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def remove(path):
    if os.path.exists(path):
        os.remove(path)
        print('removed ', path)
    else:
        print('does not exist ', path)

def move_folder(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    shutil.rmtree(src)

def insert_breakpoint(msg='break pt'):
    import sys
    sys.exit(msg)

def get_clf_report(y_true, y_pred, y_pred_prob=None):

    from sklearn.metrics import confusion_matrix, classification_report
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.metrics import roc_auc_score

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    report = classification_report(y_true, y_pred)
    print(report)
    acc = accuracy_score(y_true, y_pred)
    print('Accuracy : ', acc)

    if y_pred_prob is not None:
        prec = precision_score(y_true, y_pred)
        print('Precision : ', prec)
        recall = recall_score(y_true, y_pred)
        print('Recall : ', prec)
        auc = roc_auc_score(y_true, y_pred_prob, average='macro')
        print('AUC Score : ', auc)
        return cm, report, acc, prec, recall, auc

    return cm, report, acc


def plot_model_history(log_path, save_path, addn_info=''):

    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt
    import pandas as pd

    title = log_path.split('/')[-1]
    title += str(addn_info)

    df = pd.read_csv(log_path)
    y = df['acc']
    x = range(len(y))
    plt.plot(x, y, '-b', label='train_acc')

    y = df['val_acc']
    x = range(len(y))
    plt.plot(x, y, '-r', label='valid_acc')

    y = df['loss']
    x = range(len(y))
    plt.plot(x, y, '-g', label='train_loss')

    y = df['val_loss']
    x = range(len(y))
    plt.plot(x, y, '-m', label='val_loss')

    plt.title(title)
    plt.legend(loc='upper left')
    plt.ylim(-1.5, 2.0)

    plt.savefig(save_path)


def normalize(arr, low=0, high=255,mode='min_max'):
    if mode == 'min_max':
        diff = high - low
        if diff > 0:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
            arr = arr * diff
        return arr
    else:
        mu = np.mean(arr); sigma = np.std(arr)
        arr = (arr-mu)/sigma
        return arr

def get_count(path):
    for root, dirs, files in os.walk(path):
        count = sum([len(f) for r, d, f in os.walk(root)])
        print(root, ':', count)


def know_shape(path):
    arr = np.load(path)
    print(arr.shape)
    print(arr.dtype)


def check_file_paths(file_lists):
    file_lists = np.array(file_lists).flatten().tolist()
    valid_files = [x for x in file_lists if os.path.exists(x)]
    invalid_files = [x for x in file_lists if not os.path.exists(x)]
    print('VALID : ', len(valid_files))
    print('INVALID : ', len(invalid_files))
    return valid_files, invalid_files


def check_list_diff(a, b):

    if len(a) > len(b):
        large = set(a)
        small = set(b)
    else:
        large = set(b)
        small = set(a)

    common = large.intersection(small)
    print(str(len(common)) + ' are common')

    not_present = large.difference(small)
    print(str(len(not_present)) + ' are diff')

    return common, not_present


def write_imgs_serially(imgs, base_path='tmp/tmp_imgs/', bgr_to_rgb=False, verbose=0):

    clear_folder(base_path)
    imgs = np.array(imgs)

    for i, img in enumerate(imgs):
        name = base_path + str(i).zfill(5) + '.jpg'
        if bgr_to_rgb:
            img = img[..., ::-1]
        status = cv2.imwrite(name, img)
        if verbose > 0:
            print(name, status)
    print(str(imgs.shape[0]) + ' images written to ', base_path)


def split_train_test(X, y=None, test_size=0.20, seed=42):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    return (X_train, y_train), (X_test, y_test)


def check_array_prop(arr):

    print(type(arr))
    print(arr.shape)
    print(arr.dtype)

    print('MAX : ', np.max(arr))
    print('MIN : ', np.min(arr))
    print('MEAN : ', np.mean(arr))
    print('STD : ', np.std(arr))


def find_largest_contour_bbox(img, threshold=127):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = normalize(img, 0, 255)
    thresh_img = ((img > threshold)*255).astype(np.uint8)
    __, contours, __ = cv2.findContours(
        thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = np.array([cv2.contourArea(x) for x in contours])
    idx = np.argmax(contour_areas)
    cnt = contours[idx]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h

def randomly_select(arr,n):
    replace = True if n > len(arr) else False
    idx = np.random.choice(len(arr),size=n,replace=replace)
    return arr[idx]

if __name__ == '__main__':
    # img_path = 'data/images/Training/IDRiD_01.jpg'
    # img = cv2.imread(img_path)
    # x, y, w, h = find_largest_contour_bbox(img, threshold=50)
    # out = img[y:y+h,x:x+w,:]
    # cv2.imwrite('tmp/out.jpg', out)
    
    # a = np.random.normal(size=(5000,32,32,3))
    # randomly_select(a,n=100) 

    # a = np.arange(10)
    # check_array_prop(a)

    # a = normalize(a,0,255,'min_max')
    # a = normalize(a,0,255,'norm')
    # check_array_prop(a)

    a = np.random.randint(10,size=(500,32,32,3)) > 5
    print(a.shape)
    b = randomly_select(a,600)
    print(b.shape)
    # b = randomly_select(a,5)
    # b = randomly_select(a,10)