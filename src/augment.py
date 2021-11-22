from utils import make_env, Storage, orthogonal_init
import imageio
import torch.nn.functional as F
from random import randrange
import torch
import warnings
warnings.filterwarnings("ignore")

def stretch(frame, orgW, orgH):
    factor = (orgW / frame.shape[1] + 1e-5, orgH / frame.shape[2] + 1e-5)
    img = frame.float().unsqueeze(0)
    return F.interpolate(img, scale_factor=factor)[0,:]

def position(frame, orgW, orgH):
    zeros = torch.zeros(3, orgW, orgH)
    width = frame.shape[1]
    height = frame.shape[2]
    offsetX = randrange(orgW-width)
    offsetY = randrange(orgH-height)
    zeros[:,offsetX:offsetX+width,offsetY:offsetY+height] = frame
    return zeros



def identity(frame):
    return frame

def crop(frame):
    orgW = frame.shape[1]
    orgH = frame.shape[2]
    cropsize = randrange(32) + min(orgW, orgH) - 32
    offsetX = randrange(orgW-cropsize)
    offsetY = randrange(orgH-cropsize)
    cropped = frame[:,offsetX:offsetX+cropsize,offsetY:offsetY+cropsize]
    stretched = stretch(cropped, orgW, orgH)
    return stretched

def translate(frame):
    orgW = frame.shape[1]
    orgH = frame.shape[2]
    cropsize = randrange(32) + min(orgW, orgH) - 32
    offsetX = randrange(orgW-cropsize)
    offsetY = randrange(orgH-cropsize)
    cropped = frame[:,offsetX:offsetX+cropsize,offsetY:offsetY+cropsize]
    return position(cropped, orgW, orgH)

def cutout(frame):
    orgW = frame.shape[1]
    orgH = frame.shape[2]
    cutoutWidth = randrange(20) + 4
    cutoutHeight = randrange(20) + 4
    zeros = torch.zeros(3, cutoutWidth, cutoutHeight)
    offsetX = randrange(orgW-cutoutWidth)
    offsetY = randrange(orgH-cutoutHeight)
    frame[:,offsetX : offsetX + cutoutWidth, offsetY : offsetY + cutoutHeight] = zeros
    return frame

def colormix(frame):
    redWeight = 0.3 + randrange(100) / 100.0
    greenWeight = 0.3 + randrange(100) / 100.0
    blueWeight = 0.3 + randrange(100) / 100.0
    frame[0] = frame[0,:] * redWeight
    frame[1] = frame[1,:] * greenWeight
    frame[2] = frame[2,:] * blueWeight
    return frame

AugmentationFuncArr = []

def setAugmentationMode(mode, environments):
    global AugmentationFuncArr
    AugmentationFuncArr = []
    for i in range(environments):
        if mode == 0:
            AugmentationFuncArr.append(identity)
        elif mode == 1:
            AugmentationFuncArr.append(crop)
        elif mode == 2:
            AugmentationFuncArr.append(translate)
        elif mode == 3:
            AugmentationFuncArr.append(cutout)
        elif mode == 4:
            AugmentationFuncArr.append(colormix)

def setRandomAugmentationMode(environments):
    global AugmentationFuncArr
    AugmentationFuncArr = []
    for i in range(environments):
        mode = randrange(5)
        if mode == 0:
            AugmentationFuncArr.append(identity)
        elif mode == 1:
            AugmentationFuncArr.append(crop)
        elif mode == 2:
            AugmentationFuncArr.append(translate)
        elif mode == 3:
            AugmentationFuncArr.append(cutout)
        elif mode == 4:
            AugmentationFuncArr.append(colormix)

def augment(obs):
    for i in range(obs.shape[0]):
        obs[i] = AugmentationFuncArr[i % len(AugmentationFuncArr)](obs[i])
    return obs

def testCrop():
    env = make_env(4, num_levels=10, gamma=0.999, env_name='coinrun')
    obs = env.reset()
    for i in range(obs.shape[0]):
        frame = (obs[0,:]*255.)
        imageio.imsave("crop" + str(i) + ".png",crop(frame).T.byte())

def testTranslate():
    env = make_env(4, num_levels=10, gamma=0.999, env_name='coinrun')
    obs = env.reset()
    for i in range(obs.shape[0]):
        frame = (obs[0,:]*255.)
        imageio.imsave("translate" + str(i) + ".png",translate(frame).T.byte())

def testCutout():
    env = make_env(4, num_levels=10, gamma=0.999, env_name='coinrun')
    obs = env.reset()
    for i in range(obs.shape[0]):
        frame = (obs[0,:]*255.)
        imageio.imsave("cutout" + str(i) + ".png",cutout(frame).T.byte())

def testColorMix():
    env = make_env(4, num_levels=10, gamma=0.999, env_name='coinrun')
    obs = env.reset()
    for i in range(obs.shape[0]):
        frame = (obs[0,:]*255.)
        imageio.imsave("colorMix" + str(i) + ".png",colormix(frame).T.byte())

testCrop()