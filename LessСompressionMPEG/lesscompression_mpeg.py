import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import random

def segmentImage(anchor, blockSize=16):
    h, w = anchor.shape
    hSegments = int(h / blockSize)
    wSegments = int(w / blockSize)
    return hSegments, wSegments

def getCenter(x, y, blockSize):
    return int(x + blockSize / 2), int(y + blockSize / 2)

def getAnchorSearchArea(x, y, anchor, blockSize, searchArea):
    h, w = anchor.shape
    cx, cy = getCenter(x, y, blockSize)
    sx = max(0, cx - int(blockSize / 2) - searchArea)
    sy = max(0, cy - int(blockSize / 2) - searchArea)
    anchorSearch = anchor[sy:min(sy + searchArea * 2 + blockSize, h),
                          sx:min(sx + searchArea * 2 + blockSize, w)]
    return anchorSearch


def getBlockZone(p, aSearch, tBlock, blockSize):
    px, py = p
    px, py = px - int(blockSize / 2), py - int(blockSize / 2)
    px, py = max(0, px), max(0, py)
    aBlock = aSearch[py:py + blockSize, px:px + blockSize]

    if aBlock.shape != tBlock.shape:
        padded_block = np.zeros(tBlock.shape, dtype=aBlock.dtype)
        h_eff, w_eff = aBlock.shape
        padded_block[:h_eff, :w_eff] = aBlock
        aBlock = padded_block

    try:
        assert aBlock.shape == tBlock.shape
    except Exception as e:
        print(f"Помилка розмірності: {aBlock.shape} проти {tBlock.shape}")
        print(e)

    return aBlock

def getMAD(tBlock, aBlock):
    return np.sum(np.abs(np.subtract(tBlock, aBlock))) / (tBlock.shape[0] * tBlock.shape[1])


def getBestMatch(tBlock, aSearch, blockSize):
    step = 4
    ah, aw = aSearch.shape
    acy, acx = int(ah / 2), int(aw / 2)
    minMAD = float("+inf")
    minP = None

    while step >= 1:
        p1 = (acx, acy)
        p2 = (acx + step, acy)
        p3 = (acx, acy + step)
        p4 = (acx + step, acy + step)
        p5 = (acx - step, acy)
        p6 = (acx, acy - step)
        p7 = (acx - step, acy - step)
        p8 = (acx + step, acy - step)
        p9 = (acx - step, acy + step)
        pointList = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

        for p in range(len(pointList)):
            aBlock = getBlockZone(pointList[p], aSearch, tBlock, blockSize)
            MAD = getMAD(tBlock, aBlock)
            if MAD < minMAD:
                minMAD = MAD
                minP = pointList[p]
        step = int(step / 2)
        acx, acy = minP

    px, py = minP
    px, py = px - int(blockSize / 2), py - int(blockSize / 2)
    px, py = max(0, px), max(0, py)
    matchBlock = aSearch[py:py + blockSize, px:px + blockSize]

    if matchBlock.shape != tBlock.shape:
        padded_match = np.zeros(tBlock.shape, dtype=matchBlock.dtype)
        h_eff, w_eff = matchBlock.shape
        padded_match[:h_eff, :w_eff] = matchBlock
        matchBlock = padded_match

    return matchBlock

def blockSearchBody(anchor, target, blockSize, searchArea=7):
    h, w = anchor.shape
    hSegments, wSegments = segmentImage(anchor, blockSize)
    predicted = np.ones((h, w)) * 255
    bcount = 0
    for y in range(0, int(hSegments * blockSize), blockSize):
        for x in range(0, int(wSegments * blockSize), blockSize):
            bcount += 1
            targetBlock = target[y:y + blockSize, x:x + blockSize]
            anchorSearchArea = getAnchorSearchArea(x, y, anchor, blockSize, searchArea)
            anchorBlock = getBestMatch(targetBlock, anchorSearchArea, blockSize)
            predicted[y:y + blockSize, x:x + blockSize] = anchorBlock
    assert bcount == int(hSegments * wSegments)
    return predicted

def getResidual(target, predicted):
    return np.subtract(target, predicted)

def getReconstructTarget(residual, predicted):
    return np.add(residual, predicted)

def getBitsPerPixel(im):
    h, w = im.shape
    im_list = im.tolist()
    bits = 0
    for row in im_list:
        for pixel in row:
            bits += math.log2(abs(pixel) + 1)
    return bits / (h * w)

def getFrames(filename, first_frame, second_frame):
    cap = cv2.VideoCapture(filename)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame - 1)
    res1, fr1 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, second_frame - 1)
    res2, fr2 = cap.read()
    cap.release()
    return fr1, fr2

def main(anchorFrame, targetFrame, blockSize=16, saveOutput=True, outfile="Results_Fesyk"):
    bitsAnchor = []
    bitsDiff = []
    bitsPredicted = []
    h, w, ch = anchorFrame.shape
    print(h, w, ch)
    diffFrameRGB = np.zeros((h, w, ch))
    predictedFrameRGB = np.zeros((h, w, ch))
    residualFrameRGB = np.zeros((h, w, ch))
    restoreFrameRGB = np.zeros((h, w, ch))

    for i in range(0, 3):
        anchorFrame_c = anchorFrame[:, :, i]
        targetFrame_c = targetFrame[:, :, i]

        diffFrame = cv2.absdiff(anchorFrame_c, targetFrame_c)

        predictedFrame = blockSearchBody(anchorFrame_c, targetFrame_c, blockSize)
        residualFrame = getResidual(targetFrame_c, predictedFrame)
        reconstructTargetFrame = getReconstructTarget(residualFrame, predictedFrame)

        bitsAnchor += [getBitsPerPixel(anchorFrame_c)]
        bitsDiff += [getBitsPerPixel(diffFrame)]
        bitsPredicted += [getBitsPerPixel(residualFrame)]

        diffFrameRGB[:, :, i] = diffFrame
        predictedFrameRGB[:, :, i] = predictedFrame
        residualFrameRGB[:, :, i] = residualFrame
        restoreFrameRGB[:, :, i] = reconstructTargetFrame

    isdir = os.path.isdir(outfile)
    if not isdir:
        os.mkdir(outfile)

    if saveOutput:
        cv2.imwrite(f"{outfile}/First frame.png", anchorFrame)
        cv2.imwrite(f"{outfile}/Second frame.png", targetFrame)
        cv2.imwrite(f"{outfile}/Difference between frame.png", diffFrameRGB)
        cv2.imwrite(f"{outfile}/Prediction frame.png", predictedFrameRGB)
        cv2.imwrite(f"{outfile}/Residual frame.png", residualFrameRGB)
        cv2.imwrite(f"{outfile}/Restore frame.png", restoreFrameRGB)

    barWidth = 0.25
    fig, ax = plt.subplots(figsize=(12, 8)) # Додано щоб уникнути помилки розпакування
    P1 = [sum(bitsAnchor), bitsAnchor[0], bitsAnchor[1], bitsAnchor[2]]
    Diff = [sum(bitsDiff), bitsDiff[0], bitsDiff[1], bitsDiff[2]]
    Mpeg = [sum(bitsPredicted), bitsPredicted[0], bitsPredicted[1], bitsPredicted[2]]

    br1 = np.arange(len(P1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    plt.bar(br1, P1, color='r', width=barWidth, edgecolor='grey', label='Розмір для початкового кадру')
    plt.bar(br2, Diff, color='g', width=barWidth, edgecolor='grey', label='Розмір для різниці між кадрами')
    plt.bar(br3, Mpeg, color='b', width=barWidth, edgecolor='grey', label='Розмір для різниці з компенсацією рухів')

    plt.title(f'Ступінь стиснення = {round(sum(bitsAnchor)/sum(bitsPredicted), 2)}', fontweight='bold', fontsize=15)
    plt.ylabel('Біт на піксель', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(P1))], ['Біт/Піксель RGB', 'Біт/Піксель R', 'Біт/Піксель G', 'Біт/Піксель B'])
    plt.legend()
    plt.savefig(f'{outfile}/Гістограма кількості біт на піксель для різних варіантів кодування.png', dpi=600)

    return None

if __name__ == '__main__':
    fesyk_fr = random.randint(0, 3000)
    fesyk_frame1, fesyk_frame2 = getFrames('sample4.avi', fesyk_fr, fesyk_fr + 1)
    main(fesyk_frame1, fesyk_frame2, saveOutput=True)