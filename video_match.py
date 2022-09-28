from enum import Enum
import json
import math
import os
import re
import time
import sys

import cv2
import easyocr
import imutils
import numpy as np

# I had more success with easyocr than tesseract.
#import pytesseract
#from pytesseract import Output
#tesseractConfig = '--oem 3 --psm 6'

if len(sys.argv) != 2:
  print(f'python {sys.argv[0]} {file}')
  sys.exit(1)
if not os.path.exists(sys.argv[1]):
  print(f'{sys.argv[1]} doesn\'t exist')
  sys.exit(1)

# Utility
def secondsToTime(sec):
  return time.strftime('%H:%M:%S', time.gmtime(sec))
def frameToDebugString(frame):
  return f'{secondsToTime(math.floor(frame / fps))} = {math.floor(frame / fps)}s = {frame}f'

# Possible states to match for.
class VideoState(str, Enum):
  UNKNOWN = "Unknown"
  MAP = "Map"
  COMBAT = "Combat"
  END = "End"

# Image filename to black & white template for matching.
scriptDir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def makeTemplate(filename):
  template = cv2.imread(os.path.join(scriptDir, filename))
  template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  # Edge detection doesn't work well for GFL
  #template = cv2.Canny(template, 50, 200)
  return template

# Template images to match for. You could plausibly speed this up by removing
# one of MapRadar or MapSangvis, depending on what obstructions there are, i.e.
# dalao vtuber models. MapSangvis may also be prone to false positives.
radarTemplate = makeTemplate("MapRadar.png")
resources1Template = makeTemplate("MapResources1.png")
resources2Template = makeTemplate("MapResources2.png")
stateToTemplates = {
  VideoState.MAP: [radarTemplate, makeTemplate("MapSangvis.png")],
  VideoState.COMBAT: [makeTemplate("BattlePause.png"), makeTemplate("BattleResume.png")],
}
statesToCheck = list(stateToTemplates.keys())
previousState = VideoState.UNKNOWN
frameToState = {}
if os.path.exists(sys.argv[1] + '_frameToState.json'):
  with open(sys.argv[1] + '_frameToState.json', 'r') as f:
    frameToState = json.load(f)

# Minimum template match threshold.
threshold = 0.95
scalesToCheck = np.linspace(0.2, 1.0, 20)[::-1]

cap = cv2.VideoCapture(sys.argv[1])
fps = math.floor(cap.get(cv2.CAP_PROP_FPS))

frameIndex = 0
if len(frameToState) > 0:
  frameIndex = int(list(frameToState.keys())[-1])
  print(f'Resuming from {frameToDebugString(frameIndex)}')
  cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)

def getFrameState(frame, frameIndex):
  global scalesToCheck
  # Dict of state -> (normalized correlation, image scale)
  possibleStates = {}

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  for scale in scalesToCheck:
    resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
    # Edge detection doesn't work well for GFL
    #edged = cv2.Canny(resized, 50, 200)
    for state in statesToCheck:
      for template in stateToTemplates[state]:
        # Skip if the frame is somehow smaller than the template.
        if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
          break
        # TBH I don't know whether TM_CCORR_NORMED or TM_CCOEFF_NORMED is
        # better here.
        result = cv2.matchTemplate(resized, template, cv2.TM_CCORR_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        if maxVal > threshold and (state not in possibleStates or possibleStates[state][0] < maxVal):
          possibleStates[state] = (maxVal, scale)
          # Originally, this loop was going to short circuit, but I found
          # that that didn't really speed up this script, probably because
          # the frame manipulation is more expensive than the template
          # matching.

  # Find the most probable state. If this is the first image positively
  # matched, then also change scalesToCheck to the scale of the match to
  # speed up future frames.
  if len(possibleStates) == 0:
    return VideoState.UNKNOWN
  elif len(possibleStates) == 1:
    if len(scalesToCheck) > 1:
      scalesToCheck = [list(possibleStates.values())[0][1]]
    return list(possibleStates.keys())[0]
  else:
    maxState = VideoState.UNKNOWN
    maxValue = None
    for state, value in possibleStates.items():
      if maxValue is None or value[0] > maxValue[0]:
        maxState = state
        maxValue = value
    if len(scalesToCheck) > 1:
      scalesToCheck = [maxValue[1]]
    return maxState

while cap.isOpened():
  ret, frame = cap.read()
  if ret:
    if math.floor(frameIndex / fps) % 10 == 0:
      print(frameToDebugString(frameIndex), flush=True)
      if len(frameToState) > 0:
        # Save the current frameToState so this script can resume if interrupted
        with open(sys.argv[1] + '_frameToState.json', 'w') as f:
          f.write(json.dumps(frameToState))
    frameToState[frameIndex] = getFrameState(frame, frameIndex)
    
    # Go to the next second. You can change this to look at more or less frames.
    frameIndex += fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
  else:
    cap.release()
    break

if len(frameToState) == 0:
  print('No segments')
  sys.exit(1)

with open(sys.argv[1] + '_frameToState.json', 'w') as f:
  f.write(json.dumps(frameToState))

# Crop the score, turn, and remaining AP. This can largely be figured out from
# where the radar and resources are.
def cropScore(resized, radarMaxLoc, resources2MaxLoc):
  x1 = int(radarMaxLoc[0] + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.495)
  x2 = int(x1 + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.105)
  y1 = int(resources2MaxLoc[1] + resources2Template.shape[0] * 0.2)
  y2 = int(y1 + resources2Template.shape[0] * 0.45)
  #print("score", resized.shape, radarMaxLoc, resources2MaxLoc, resources2Template.shape, x1, x2, y1, y2)
  return resized[y1:y2, x1:x2]
def cropTurn(resized, radarMaxLoc, resources2MaxLoc):
  x1 = int(radarMaxLoc[0] + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.35)
  x2 = int(x1 + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.06)
  y1 = int(resources2MaxLoc[1] + resources2Template.shape[0] * 0.2)
  y2 = int(y1 + resources2Template.shape[0] * 0.75)
  #print("turn", resized.shape, radarMaxLoc, resources2MaxLoc, resources2Template.shape, x1, x2, y1, y2)
  return resized[y1:y2, x1:x2]
def cropAp(resized, radarMaxLoc, resources2MaxLoc):
  x1 = int(resources2MaxLoc[0] - resources2Template.shape[1] * 4)
  x2 = int(x1 + resources2Template.shape[1] * 3)
  y1 = int(radarMaxLoc[1] + resources2Template.shape[0] * 2)
  y2 = int(y1 + resources2Template.shape[0] * 0.9)
  #print("AP", resized.shape, radarMaxLoc, resources2MaxLoc, resources2Template.shape, x1, x2, y1, y2)
  return resized[y1:y2, x1:x2]
def getMapStateFromFrame(frame, frameIndex, reader):
  # If the segments are being computed with a complete frameToState, then
  # scalesToCheck isn't set correctly. We run getFrameState once to just
  # set scaling.
  global scalesToCheck
  if len(scalesToCheck) > 1:
    getFrameState(frame, frameIndex)

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  resized = imutils.resize(gray, width = int(gray.shape[1] * scalesToCheck[0]))
  radarResult = cv2.matchTemplate(resized, radarTemplate, cv2.TM_CCORR_NORMED)
  (_, radarMaxVal, _, radarMaxLoc) = cv2.minMaxLoc(radarResult)
  resources2Result = cv2.matchTemplate(resized, resources2Template, cv2.TM_CCORR_NORMED)
  (_, resources2MaxVal, _, resources2MaxLoc) = cv2.minMaxLoc(resources2Result)
  if radarMaxVal < threshold or resources2MaxVal < threshold:
    #print(f'Can\'t match map items at {frameToDebugString(int(frameIndex))}')
    return None

  scoreProcessed = cropScore(resized, radarMaxLoc, resources2MaxLoc)
  #scoreProcessed = cv2.filter2D(scoreCropped, -1, laplacianFilter)
  #scoreProcessed = cv2.threshold(scoreProcessed, 20, 150, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  #cv2.imwrite(sys.argv[1] + f'_{frameIndex}_score.jpg', scoreProcessed)
  #scoreOcrString = pytesseract.image_to_string(scoreProcessed, config=tesseractConfig).strip()
  #print(scoreOcrString)
  scoreOcrData = reader.readtext(scoreProcessed, detail=0)
  scoreOcrString = "".join(scoreOcrData)
  if scoreOcrString == "" or not scoreOcrString.isnumeric():
    return None
  score = int(scoreOcrString)

  turnProcessed = cropTurn(resized, radarMaxLoc, resources2MaxLoc)
  # Blast the image because Sijun loves putting his music overlay on top of the turn # display.
  turnProcessed = cv2.threshold(turnProcessed, 220, 255, cv2.THRESH_BINARY)[1]
  #turnOcrString = pytesseract.image_to_string(turnProcessed, config=tesseractConfig).strip()
  turnOcrData = reader.readtext(turnProcessed, detail=0)
  turnOcrString = "".join(turnOcrData)
  # Strip leading 0. Sometimes the "0" with a slash is OCR'd as "2", "8", or "Z".
  turnOcrString = re.sub(r'^[02389]+(.+)', r'\1', re.sub(r'[iI]', '1', re.sub(r'[oOsSzZ]', '0', turnOcrString)))
  #cv2.imwrite(sys.argv[1] + f'_{frameIndex}_turn.jpg', turnProcessed)
  if turnOcrString == "" or not turnOcrString.isnumeric():
    turn = None
  else:
    turn = int(turnOcrString)

  # For the AP count we look at the yellow parts of the image.
  # Technically we should probably check that the frame isn't in planning mode, but
  # since this function should
  hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  hsvResized = imutils.resize(hsv, width = int(hsv.shape[1] * scalesToCheck[0]))
  lowerYellow = np.array([10,50,100])
  upperYellow = np.array([40,255,255])
  mask = cv2.inRange(hsvResized, lowerYellow, upperYellow)
  hsvResized = cv2.bitwise_and(hsvResized, hsvResized, mask=mask)
  apProcessed = cropAp(hsvResized, radarMaxLoc, resources2MaxLoc)
  apProcessed = cv2.cvtColor(apProcessed, cv2.COLOR_BGR2GRAY)
  apProcessed = cv2.threshold(apProcessed, 100, 255, cv2.THRESH_BINARY)[1]
  #apOcrString = pytesseract.image_to_string(apProcessed, config=tesseractConfig).strip()
  apOcrData = reader.readtext(apProcessed, detail=0)
  apOcrString = "".join(apOcrData)
  # Strip leading 0 (or "o" or "O").
  apOcrString = re.sub(r'^0+(.+)', r'\1', re.sub(r'[iI]', '1', re.sub(r'[oO]', '0', apOcrString)))
  #cv2.imwrite(sys.argv[1] + f'_{frameIndex}_ap__{apOcrString}.jpg', apProcessed)
  if apOcrString == "" or not apOcrString.isnumeric():
    ap = None
  else:
    ap = int(apOcrString)

  mapState = {
    "turn": turn,
    "score": score,
    "ap": ap
  }
  #print(f'{frameToDebugString(int(frameIndex))}: {mapState} ({apOcrString})', flush=True)
  return mapState

blankMapState = {
  "turn": "",
  "score": "",
  "ap": ""
}
def generateSegments(frameToState, cap, reader):
  segments = []
  mapSegmentFrames = []
  segmentStartFrame = int(list(frameToState.keys())[0])
  segmentState = list(frameToState.values())[0]
  def processFrame(frameIndex, state):
    nonlocal segments, mapSegmentFrames, segmentStartFrame, segmentState
    if state != VideoState.UNKNOWN and state != segmentState:
      startMapState = None
      endMapState = None
      if segmentState == VideoState.MAP and cap.isOpened():
        for f in mapSegmentFrames[:60]:
          cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
          ret, frame = cap.read()
          if not ret:
            break
          startMapState = getMapStateFromFrame(frame, int(f), reader)
          if startMapState is not None:
            break
        for f in reversed(mapSegmentFrames[-60:]):
          cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
          ret, frame = cap.read()
          if not ret:
            break
          endMapState = getMapStateFromFrame(frame, int(f), reader)
          if endMapState is not None and (startMapState is None or endMapState["score"] >= startMapState["score"]):
            break
      if startMapState is None:
        startMapState = blankMapState
      if endMapState is None:
        endMapState = blankMapState

      segments.append((segmentStartFrame / fps, int(frameIndex) / fps, segmentState, startMapState, endMapState))

      segmentStartFrame = int(frameIndex)
      segmentState = state
      mapSegmentFrames = []
    if state == VideoState.MAP:
      mapSegmentFrames.append(frameIndex)

  for frame, state in frameToState.items():
    processFrame(frame, state)
  processFrame(int(list(frameToState.keys())[-1]), VideoState.END)
  return segments

cap = cv2.VideoCapture(sys.argv[1])
reader = easyocr.Reader(['en'])
segments = generateSegments(frameToState, cap, reader)

#with open(sys.argv[1] + '_segments.json', 'w') as f:
#  f.write(json.dumps(segments))
with open(sys.argv[1] + '_segments.tsv', 'w') as f:
  header = '\t'.join(["Turn Start", "AP Start", "Turn End", "AP End", "Score Start", "Score End", "Start", "End", "Duration (s)", "State"])
  print(header, flush=True)
  f.write(header + '\n')
  for segment in segments:
    line = '\t'.join(str(x) for x in [
      segment[3]["turn"],
      segment[3]["ap"],
      segment[4]["turn"],
      segment[4]["ap"],
      segment[3]["score"],
      segment[4]["score"],
      secondsToTime(segment[0]),
      secondsToTime(segment[1]),
      segment[1] - segment[0],
      segment[2]
    ])
    print(line, flush=True)
    f.write(line + '\n')

