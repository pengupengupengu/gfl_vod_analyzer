from enum import Enum
import re
import os
import time

import cv2
import numpy as np

class OcrTargets(str, Enum):
  SCORE = "score"
  TURN = "turn"
  AP_LEFT = "ap"
  def __str__(self):
    return str(self.value)

MIN_WIDTH = {
  OcrTargets.SCORE: 4,
  OcrTargets.TURN: 7,
  OcrTargets.AP_LEFT: 16,
}
MIN_HEIGHT = {
  OcrTargets.SCORE: 16,
  OcrTargets.TURN: 40,
  OcrTargets.AP_LEFT: 42,
}
MAX_HEIGHT = {
  OcrTargets.SCORE: 28,
  OcrTargets.TURN: 60,
  OcrTargets.AP_LEFT: 66,
}

class GflOcr:
  def __init__(self, scriptDir, debug=False, enableKnn=False):
    self.debug = debug
    self.debugImageIndex = 0
    self.knnModels = {}
    self.responseToLabelDicts = {}
    for target in OcrTargets:
      samples = np.empty((0, 100))
      responses = []
      labelToResponse = {}
      imageDirs = ["_"] + [str(x) for x in range(0, 10)] + ["combo"]
      for x in imageDirs:
        #print(x)
        imageDir = os.path.join(scriptDir, "training_data", target, str(x))
        for file in os.listdir(imageDir):
          if ".png" not in file:
            continue
          if x == "combo":
            match = re.search(r'^(\d+)_', file)
            if match is None:
              continue
            label = match[1]
          else:
            label = x
          sample = cv2.imread(os.path.join(imageDir, file))
          sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
          sample = cv2.resize(sample, (10, 10))
          sample = sample.reshape((1, 100))
          samples = np.append(samples, sample, 0)
          if label not in labelToResponse:
            labelToResponse[label] = len(labelToResponse)
          responses.append(labelToResponse[label])
      samples = samples.astype(np.float32)
      responses = np.array(responses, np.int32)
      responses = responses.reshape((responses.size, 1))
      self.knnModels[target] = cv2.ml.KNearest_create()
      self.knnModels[target].train(samples, cv2.ml.ROW_SAMPLE, responses)
      self.responseToLabelDicts[target] = dict([(v, k) for k, v in labelToResponse.items()])

  def getDebugImageSuffix(self):
    self.debugImageIndex = self.debugImageIndex + 1
    return str(int(time.time()))[-7:-1] + str(self.debugImageIndex % 1000).rjust(3, "0")

  def recognize(self, imgGray, ocrTarget, debugPrefix=""):
    #imgGray = 255 - cv2.threshold(imgGray, 100, 255, cv2.THRESH_BINARY_INV)[1]
    contourTarget = cv2.adaptiveThreshold(imgGray, 255, 1, 1, 11, 2)
    if self.debug:
      imgGrayAnnotated = imgGray
      imgGrayAnnotated = cv2.cvtColor(imgGrayAnnotated, cv2.COLOR_GRAY2BGR)
      contourTargetAnnotated = cv2.cvtColor(contourTarget, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(contourTarget, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    recognizedNumbersByXPos = {}
    
    boundingBoxes = []
    predictionInputs = np.empty((0, 100))
    for contour in contours:
      [x, y, w, h] = cv2.boundingRect(contour)
      if w < MIN_WIDTH[ocrTarget]:# or w > MAX_WIDTH[ocrTarget]:
        continue
      if h < MIN_HEIGHT[ocrTarget] or h > MAX_HEIGHT[ocrTarget]:
        continue
      #print([x, y, w, h])
      if self.debug:
        cv2.rectangle(imgGrayAnnotated, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(contourTargetAnnotated, (x, y), (x+w, y+h), (0, 0, 255), 1)
      cropped = imgGray[y:y+h, x:x+w]
      predictionInput = cv2.resize(cropped, (10, 10))
      predictionInput = predictionInput.reshape((1, 100))
      #predictionInput = np.float32(predictionInput)

      boundingBoxes.append([x, y, w, h])
      predictionInputs = np.append(predictionInputs, predictionInput, 0)
    if len(boundingBoxes) == 0:
      return (None, None)

    _, _, neighborResponses, distances = self.knnModels[ocrTarget].findNearest(np.float32(predictionInputs), k=5 if self.debug else 1)

    for [x, y, w, h], neighResp, dists in zip(boundingBoxes, neighborResponses, distances):
      label = self.responseToLabelDicts[ocrTarget][int(neighResp[0])]
      #print(x, label, [self.responseToLabelDicts[ocrTarget][x] for x in neighResp], dists)
      if dists[0] > 1e6:
        if self.debug:
          cv2.imwrite(f'training_data/{ocrTarget}/_/new/__{self.getDebugImageSuffix()}.png', imgGray[y:y+h, x:x+w])
        continue
      if label != "_":
        recognizedNumbersByXPos[x] = label

      if self.debug:
        if len(label) > 1:
          directory = "combo"
          prefix = "combo_" + label
        else:
          directory = label
          prefix = label
        if len(label) == 1 and any([x != neighResp[0] for x in neighResp[1:]]):
          cv2.imwrite(f'training_data/{ocrTarget}/conflict/new/__{self.getDebugImageSuffix()}.png', imgGray[y:y+h, x:x+w])
        else:
          cv2.imwrite(f'training_data/{ocrTarget}/{directory}/new/{prefix}_{self.getDebugImageSuffix()}.png', imgGray[y:y+h, x:x+w])
        cv2.rectangle(imgGrayAnnotated, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(imgGrayAnnotated, label, (x + 1, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))
        cv2.rectangle(contourTargetAnnotated, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.putText(contourTargetAnnotated, label, (x + 1, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0))
    if self.debug:
      cv2.imwrite(f'test_images/debug/{debugPrefix}_imgGrayAnnotated.png', imgGrayAnnotated)
      cv2.imwrite(f'test_images/debug/{debugPrefix}_contourTargetAnnotated.png', contourTargetAnnotated)
    if len(recognizedNumbersByXPos) == 0:
      return (None, None)
    resultString = "".join([str(v) for k, v in sorted(recognizedNumbersByXPos.items(), key=lambda x: x[0])])
    if ocrTarget == OcrTargets.TURN or ocrTarget == OcrTargets.AP_LEFT:
      if len(resultString) != 2:
        return (None, None)
      resultNum = int(re.sub(r'^0', '', resultString))
    else:
      if resultString[0] == "0" and resultString != "0":
        return (None, None)
      resultNum = int(resultString)
    return (resultNum, resultString)

if __name__ == "__main__":
  scriptDir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  print("Testing...")
  gflOcr = GflOcr(scriptDir, debug=True)
  for target in OcrTargets:
    imageDir = os.path.join(scriptDir, "test_images", target)
    for file in os.listdir(imageDir):
      if ".png" not in file:
        continue
      match = re.search(r'^(\d+)__', file)
      if match is None:
        continue
      img = cv2.imread(os.path.join(imageDir, file))
      imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      result, resultString = gflOcr.recognize(imgGray, target, debugPrefix=file)
      if resultString != match[1]:
        partialPath = os.path.join("test_images", target, file)
        print(f'  "{match[1]}" != "{resultString}" failed for "{partialPath}"')
