import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
from scipy import signal
from sklearn.linear_model import Lasso

# load image
lambdas = np.logspace(-7, 2, 10)
NUM_CV_FOLDS = 20

# basis vector matrix
def basis(u, v, x, y, P, Q):
    alpha = np.sqrt(1/P) if u == 0 else np.sqrt(2/P)
    beta = np.sqrt(1/Q) if v == 0 else np.sqrt(2/Q)
    return alpha*beta*math.cos(math.pi*(2*x-1)*(u-1)/(2*P))*math.cos(math.pi*(2*y-1)*(v-1)/(2*P))

def getBasisVectorMatrix(P,Q):
    basisVectorMatrix = np.zeros(((P*Q)*(P*Q)))
    i = 0
    for x in range(1,P+1):
        for y in range(1,Q+1):
            for v in range(1,Q+1):
                for u in range(1,P+1):
                    basisVectorMatrix[i] = basis(u, v, x, y, P, Q)
                    i += 1

    basisVectorMatrix = basisVectorMatrix.reshape((P*Q),(P*Q))
    return basisVectorMatrix

def createSnChip(chip, s, k):
    totalPixels = chip.shape[0] * chip.shape[1] 
    idx_to_keep = np.random.choice(totalPixels,s, replace=False)
    snChip = np.full(chip.shape, np.NaN)
    for i in idx_to_keep:
        x = i // k
        y = i-(x*k)
        snChip[x][y] = chip[x][y]
    return snChip

def corruptImg(img, s, k, plot=False):    
    corrupted_img = np.full((img.shape[0],img.shape[1]), np.NaN)
    for x in range(0,img.shape[0],k):
        for y in range(0,img.shape[1],k):           
            chip = img[x:x+k,y:y+k]
            corrupted_img[x:x+k,y:y+k] = createSnChip(chip, s, k)
    if plot:
        fig,ax = plt.subplots()
        ax.imshow(corrupted_img, cmap='gray')
        ax.set_title(f"corrupted image s={s}")
        plt.show()
    return corrupted_img

# given corrupted chip return basisVectorMatrix and noNan Chip
def createUnderdeterminedSystem(chip, basisVectorMatrix):
    sensed_idx= ~np.isnan(chip)
    chip_noNan = chip[sensed_idx]

    underdeterminedBVM = np.zeros((chip_noNan.shape[0],basisVectorMatrix.shape[1]))
    i = 0
    p2 = sensed_idx.shape[0]
    q2 = sensed_idx.shape[1]
    for x in range(p2):
        for y in range(q2):
            if sensed_idx[x][y]:
                underdeterminedBVM[i] = basisVectorMatrix[x*p2 + y]
                i+=1
    return underdeterminedBVM, chip_noNan

def makeTrainTestImg(src, m):
    trainimg = src.copy()
    testimg = np.full(src.shape, np.NaN)
    nonNans = np.argwhere(~np.isnan(src))
    testIdxs = np.random.choice(nonNans.shape[0],m, replace=False)
    for i in testIdxs:
        x = nonNans[i][0]
        y = nonNans[i][1]
        trainimg[x][y] = np.NaN
        testimg[x][y] = src[x][y]
   
    return trainimg, testimg

# given corruptedChip (noNans)
def CrossvalidateLambda(chip, basisVectorMatrix, m, plot=False):
    if plot:
        fig, ax = plt.subplots()
        ax.set_xscale("log")
        ax.set_title(f"MSE vs log(lambda) s={math.ceil(m*6 / 10) * 10}")
        ax.set_xlabel("log(lambda)")
        ax.set_ylabel("MSE")

    avgMSEs = []
    for fold in range(NUM_CV_FOLDS):
        chip_train, chip_test  = makeTrainTestImg(chip, m)
        chip_train_BVM, chip_train_noNAN = createUnderdeterminedSystem(chip_train, basisVectorMatrix)
        chip_test_BVM, chip_test_noNAN = createUnderdeterminedSystem(chip_test, basisVectorMatrix)
        MSEs = []
        for l in lambdas:
            model = Lasso(l).fit(chip_train_BVM, chip_train_noNAN)

            chip_test_pred = np.reshape(model.predict(chip_test_BVM), chip_test_noNAN.shape)
            MSEs.append(np.mean((chip_test_noNAN-chip_test_pred)**2))

        # plot one fold's worth of MSEs
        avgMSEs.append(MSEs)

        if plot:
            ax.plot(lambdas, MSEs, linestyle="--",alpha=0.5)

    # plot avg over all the folds
    avgMSEs = np.mean(avgMSEs, axis=0)
    optimalLambda = lambdas[np.argmin(avgMSEs)]
    if plot:
        ax.plot(lambdas, avgMSEs,c="black",label="Average MSE")
        ax.axvline(optimalLambda, label=f"Optimal Lambda ({optimalLambda})")
        ax.legend()
        plt.show()
    return optimalLambda

def reconstruct_chip(basisVectorMatrix, corrupted_chip, m, plot=False):
    optimalLambda = CrossvalidateLambda(corrupted_chip, basisVectorMatrix, m, plot=plot)
    chip_BVM, chip_noNAN = createUnderdeterminedSystem(corrupted_chip, basisVectorMatrix)
    model = Lasso(optimalLambda).fit(chip_BVM, chip_noNAN)
    # weights = np.reshape(model.coef_, (model.coef_.shape[0],1))
    pred_chip = np.reshape(model.predict(basisVectorMatrix), (corrupted_chip.shape[0], corrupted_chip.shape[1]))
    # if plot:
    #     fig, ax = plt.subplots()
    #     ax.stem(weights)
    #     ax.set_title(f"weights \nlambda={optimalLambda}")
    #     plt.show()

    return pred_chip, optimalLambda

def plotChips(original_chip, corrupted_chip, pred_chip, plot=False):
    mse = np.mean((original_chip-pred_chip)**2)/(original_chip.shape[0]*original_chip.shape[1])
    if plot:
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(corrupted_chip, cmap='gray')
        ax[0].set_title(f"corrupted image")
        ax[1].imshow(pred_chip, cmap='gray')
        ax[1].set_title(f"predicted chip \ns=30 mse={mse.round(2)}")
        plt.show()

def reconstruct_img(corrupted_img, m, k):  
    reconstructed_img = np.full((corrupted_img.shape[0],corrupted_img.shape[1]), np.NaN)
    lambdas = []
    basisVectorMatrix = getBasisVectorMatrix(k, k)
    for x in range(0,corrupted_img.shape[0],k):
        for y in range(0,corrupted_img.shape[1],k):
            corrupted_chip = corrupted_img[x:x+k,y:y+k]
            pred_chip, optimal_lambda = reconstruct_chip(basisVectorMatrix, corrupted_chip, m)
            lambdas.append(optimal_lambda)
            reconstructed_img[x:x+k,y:y+k] = pred_chip
        
    return reconstructed_img, lambdas

def simulate_reconstruct_img(img, s, k, plot=False):
    print("s:",s)
    corrupted_img = corruptImg(img, s, k, plot=plot)
    reconstructed_img = reconstruct_img(corrupted_img, s//6, k)

    mse = np.mean((img-reconstructed_img)**2)/(img.shape[0]*img.shape[1])
    if plot:
        fig,ax = plt.subplots()
        ax.imshow(reconstructed_img, cmap='gray')
        ax.set_title(f"reconstructed image mse:{mse}")
        plt.show()
    return reconstructed_img

def addMedianFilter(original_img, reconstructed_img,filterSize, plot=False):
    reconstructed_img_medfilt = signal.medfilt2d(reconstructed_img, kernel_size=filterSize)
    if plot:
        mse = np.mean((original_img-reconstructed_img_medfilt)**2)
        fig,ax = plt.subplots()
        ax.imshow(reconstructed_img_medfilt, cmap='gray')
        ax.set_title(f"reconstructed image with median filtering mse:{mse}")
        plt.show()
    return reconstructed_img_medfilt

if __name__ == "__main__":
    # load image
    k = 8
    lambdas = np.logspace(-7, 2, 10)
    M = 20

    boat = np.asarray(Image.open('fishing_boat.bmp'), dtype=np.float64)
    fig,ax = plt.subplots()
    ax.imshow(boat, cmap='gray')
    ax.set_title("original image")
    plt.show()

    savedImgs = {}
    for s in [10,20,30,40,50]:
        reconstructed_img = simulate_reconstruct_img(boat, s, k, plot=True)
        savedImgs[s] = reconstructed_img