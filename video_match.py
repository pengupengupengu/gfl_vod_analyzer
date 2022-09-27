from enum import Enum
import json
import math
import os
import time
import sys

import cv2
import imutils
import numpy as np

if len(sys.argv) != 2:
  print(f'python {sys.argv[0]} {file}')
  sys.exit(1)
if not os.path.exists(sys.argv[1]):
  print(f'{sys.argv[1]} doesn\'t exist')
  sys.exit(1)

# Utility
def secondsToTime(sec):
  return time.strftime('%H:%M:%S', time.gmtime(sec))

# Possible states to match for.
class VideoState(str, Enum):
  UNKNOWN = "Unknown"
  MAP = "Map"
  COMBAT = "Combat"

# Image filename to black & white template for matching.
scriptDir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
def makeTemplate(filename):
  template = cv2.imread(os.path.join(scriptDir, filename))
  template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  # Edge detection doesn't work well for GFL
  #template = cv2.Canny(template, 50, 200)
  return template

# Template images to match for. You could plausibly speed this up by moving
stateToTemplates = {
  VideoState.MAP: [makeTemplate("MapRadar.png"), makeTemplate("MapSangvis.png")],
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
  print(f'Resuming from {secondsToTime(math.floor(frameIndex / fps))} = {math.floor(frameIndex / fps)}s = {frameIndex}f')
  cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)

while cap.isOpened():
  ret, frame = cap.read()
  if ret:
    if math.floor(frameIndex / fps) % 10 == 0:
      print(f'{secondsToTime(math.floor(frameIndex / fps))} = {math.floor(frameIndex / fps)}s = {frameIndex}f', flush=True)
      if len(frameToState) == 0:
        # Save the current frameToState so this script can resume if interrupted
        with open(sys.argv[1] + '_frameToState.json', 'w') as f:
          f.write(json.dumps(frameToState))

    # Dict of state -> (normalized correlation, image scale)
    possibleStates = {}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for scale in scalesToCheck:
      resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
      r = gray.shape[1] / float(resized.shape[1])
      # Edge detection doesn't work well for GFL
      #edged = cv2.Canny(resized, 50, 200)
      for state in statesToCheck:
        for template in stateToTemplates[state]:
          # Skip if the frame is somehow smaller than the template.
          if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
            break
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
      previousState = VideoState.UNKNOWN
    elif len(possibleStates) == 1:
      previousState = list(possibleStates.keys())[0]
      if len(scalesToCheck) > 1:
        scalesToCheck = [list(possibleStates.values())[0][1]]
    else:
      maxState = VideoState.UNKNOWN
      maxValue = None
      for state, value in possibleStates.items():
        if maxValue is None or value[0] > maxValue[0]:
          maxState = state
          maxValue = value
      previousState = maxState
      if len(scalesToCheck) > 1:
        scalesToCheck = [maxValue[1]]
    frameToState[frameIndex] = previousState
    
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

segments = []
segmentStartFrame = int(list(frameToState.keys())[0])
segmentState = list(frameToState.values())[0]
for frame, state in frameToState.items():
  if state != VideoState.UNKNOWN and state != segmentState:
    segments.append((segmentStartFrame / fps, int(frame) / fps, segmentState))
    segmentStartFrame = int(frame)
    segmentState = state
segments.append((segmentStartFrame / fps, int(list(frameToState.keys())[-1]) / fps, segmentState))

#with open(sys.argv[1] + '_segments.json', 'w') as f:
#  f.write(json.dumps(segments))
with open(sys.argv[1] + '_segments.tsv', 'w') as f:
  header = '\t'.join(["Start", "End", "Start", "End", "Duration", "State"])
  print(header, flush=True)
  f.write(header + '\n')
  for segment in segments:
    line = '\t'.join(str(x) for x in [
      segment[0],
      segment[1],
      secondsToTime(segment[0]),
      secondsToTime(segment[1]),
      segment[1] - segment[0],
      segment[2]
    ])
    print(line, flush=True)
    f.write(line + '\n')

