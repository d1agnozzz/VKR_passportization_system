import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt
from pathlib import Path


def calibrate(showPics=True):
    calibrationDir = Path(
        "/media/ubuntuser/bf3469a2-e299-475d-b7fe-3ccd74cf24de/klepolin/Study/Diploma/checkerboard_calibration/poco_f3_wide"
    )
    imgPathList = glob.glob(os.path.join(calibrationDir, "*.jpg"))

    nRows = 13
    nCols = 8
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 1000, 0)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList = []
    imgPtsList = []

    for curImgPath in imgPathList:
        print(curImgPath)
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(
            imgGray, (nRows, nCols), None
        )

        if cornersFound:
            print('Corners found!')
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(
                imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria
            )
            imgPtsList.append(cornersRefined)
            if showPics:
                cv.drawChessboardCorners(
                    imgBGR, (nRows, nCols), cornersRefined, cornersFound
                )
                imgBGR = cv.resize(imgBGR, (1920, 1080))
                cv.imshow("Chessboard", imgBGR)
                cv.waitKey(500)
    cv.destroyAllWindows()

    repError, camMatrix, distCoeff, rvecs, tvecs = cv.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    print("Camera Matrix:\n", camMatrix)
    print(f'fx: {camMatrix[0,0]}')
    print(f'fy: {camMatrix[1,1]}')
    print(f'cx: {camMatrix[0,2]}')
    print(f'cy: {camMatrix[1,2]}')
    print(f"Dist coeff: {distCoeff}")
    print(f'K1: {distCoeff[0,0]}')
    print(f'K2: {distCoeff[0,1]}')
    print(f'P1: {distCoeff[0,2]}')
    print(f'P2: {distCoeff[0,3]}')
    print(f'K3: {distCoeff[0,4]}')
    print("Reproj Error (pixels): {:.4f}".format(repError))

    saveDir = calibrationDir
    paramPath = os.path.join(saveDir, "calibration.npz")
    np.savez(
        paramPath,
        repError=repError,
        camMatrix=camMatrix,
        distCoeff=distCoeff,
        rvecs=rvecs,
        tvecs=tvecs,
    )
    return camMatrix, distCoeff


def removeDistortion(camMatrix, distCoeff):
    imgPath = Path(
        "/home/klepolin/Study/Diploma/checkerboard_calibration/edge1.jpg"
    )
    img = cv.imread(str(imgPath))
    height, width = img.shape[:2]
    camMatrixNew, roi = cv.getOptimalNewCameraMatrix(
        camMatrix, distCoeff, (width, height), 1, (width, height)
    )
    imgUndist = cv.undistort(img, camMatrix, distCoeff, None, camMatrixNew)
    cv.imwrite('./undist.png', imgUndist)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()


def runCalibration():
    calibrate()


def runRemoveDistortion():
    camMatrix, distCoeff = calibrate()
    removeDistortion(camMatrix, distCoeff)


if __name__ == "__main__":
    runCalibration()
    # runRemoveDistortion()
