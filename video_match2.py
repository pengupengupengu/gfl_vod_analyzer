from enum import Enum
import json
import math
import os
import re
import sys
import time

import cv2
import imutils
import numpy as np

import ocr

# Utility
def secondsToTime(sec):
  return time.strftime('%H:%M:%S', time.gmtime(sec))
def frameToDebugString(fps, frameIndex):
  return f'{secondsToTime(math.floor(frameIndex / fps))} = {math.floor(frameIndex / fps)}s = {frameIndex}f'

# Possible states to match for.
class VideoState(str, Enum):
  UNKNOWN = "Unknown"
  MAP = "Map"
  COMBAT = "Combat"
  LOADING = "Loading"
  def __str__(self):
    return str(self.value)

# Crop the score, turn, and remaining AP. This can largely be figured out from
# where the radar and resources are.
def getScoreLocation(radarTemplate, radarMaxLoc, resources2Template, resources2MaxLoc):
  x1 = int(radarMaxLoc[0] + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.495)
  x2 = int(x1 + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.12)
  y1 = int(resources2MaxLoc[1] + resources2Template.shape[0] * 0.2)
  y2 = int(y1 + resources2Template.shape[0] * 0.45)
  return ((x1, y1), (x2, y2))
def getTurnLocation(radarTemplate, radarMaxLoc, resources2Template, resources2MaxLoc):
  x1 = int(radarMaxLoc[0] + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.35)
  x2 = int(x1 + (resources2MaxLoc[0] - radarMaxLoc[0]) * 0.06)
  y1 = int(resources2MaxLoc[1] + resources2Template.shape[0] * 0.2)
  y2 = int(y1 + resources2Template.shape[0] * 0.75)
  return ((x1, y1), (x2, y2))
def getApLocation(radarTemplate, radarMaxLoc, resources2Template, resources2MaxLoc):
  x1 = int(resources2MaxLoc[0] - resources2Template.shape[1] * 4)
  x2 = int(x1 + resources2Template.shape[1] * 3)
  y1 = int(radarMaxLoc[1] + resources2Template.shape[0] * 2)
  y2 = int(y1 + resources2Template.shape[0] * 0.9)
  return ((x1, y1), (x2, y2))
def crop(img, location):
  return img[location[0][1]:location[1][1], location[0][0]:location[1][0]]
def processScoreImage(scoreImg):
  scoreProcessed = scoreImg
  scoreProcessed = 255 - cv2.threshold(scoreProcessed, 120, 255, cv2.THRESH_BINARY_INV)[1]
  return scoreProcessed
def processTurnImage(turnImg):
  turnProcessed = turnImg
  turnProcessed = 255 - cv2.threshold(turnProcessed, 180, 255, cv2.THRESH_BINARY)[1]
  return turnProcessed
def processApImage(apImg):
  hsv = cv2.cvtColor(apImg, cv2.COLOR_BGR2HSV)
  lowerYellow = np.array([10, 50, 100])
  upperYellow = np.array([40, 255, 255])
  mask = cv2.inRange(hsv, lowerYellow, upperYellow)
  apProcessed = cv2.bitwise_and(hsv, hsv, mask=mask)
  apProcessed = cv2.cvtColor(apProcessed, cv2.COLOR_BGR2GRAY)
  apProcessed = 255 - cv2.threshold(apProcessed, 100, 255, cv2.THRESH_BINARY)[1]
  return apProcessed

STARTING_GRANULARITY = 16.0
MINIMUM_GRANULARITY = 1.0
#STARTING_GRANULARITY = 32.0
#MINIMUM_GRANULARITY = 32.0

class VideoParser:
  def __init__(self, scriptDir, videoFilePath, matchThreshold=0.95):
    self.scriptDir = scriptDir
    self.gflOcr = ocr.GflOcr(scriptDir)
    self.videoFilePath = videoFilePath
    self.jsonFilePath = videoFilePath + "_parserState.json"
    self.videoCap = cv2.VideoCapture(videoFilePath)
    if self.videoCap is None:
      raise Exception(f'Failed to open {videoFilePath}')
    self.fps = self.videoCap.get(cv2.CAP_PROP_FPS)
    self.totalFrames = math.floor(self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    self.totalDurationString = secondsToTime(math.floor(self.totalFrames / self.fps))
    self.state = {
      "matchThreshold": matchThreshold,
      "scale": -1,
      "scoreLoc": None,
      "turnLoc": None,
      "apLoc": None
    }
    self.segmentGroups = None
    if os.path.exists(self.jsonFilePath):
      with open(self.jsonFilePath, 'r') as f:
        self.state = json.load(f)
    
  def save(self):
    with open(self.jsonFilePath, 'w') as f:
      f.write(json.dumps(self.state))
    
  def makeTemplates(self):
    def makeTemplate(filename):
      template = cv2.imread(os.path.join(self.scriptDir, filename))
      template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
      return template
    # Template images to match for.
    self.radarTemplate = makeTemplate("MapRadar.png")
    self.resources2Template = makeTemplate("MapResources2.png")
    self.battlePauseTemplate = makeTemplate("BattlePause.png")
    self.loadingTemplate = makeTemplate("Loading.png")

  def getFrame(self, frameIndex):
    self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, int(frameIndex))
    ret, frame = self.videoCap.read()
    if not ret:
      return None
    return frame

  def getResizedFrame(self, frameIndex):
    frame = self.getFrame(frameIndex)
    if frame is None:
      return None
    if self.state["scale"] <= 0:
      raise Exception(f'Invalid scale')
    return imutils.resize(frame, width = int(frame.shape[1] * self.state["scale"]))

  def calibrateScaling(self):
    if self.state["scale"] > 0:
      return
    print('Calibrating scaling...')
    scaleToMaxVal = {}
    templateToCheck = [self.radarTemplate, self.resources2Template, self.battlePauseTemplate, self.loadingTemplate]
    possibleScales = []
    for frameIndex in range(0, self.totalFrames, math.floor(self.fps * 60)):
      if len(possibleScales) >= 7:
        break
      print(f'  {secondsToTime(math.floor(frameIndex / self.fps))} / {self.totalDurationString}...')

      frame = self.getFrame(frameIndex)
      if frame is None:
        continue
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Try to find the scale from a wide range at first, then narrow it to successive candidates.
      scalesToCheck = []
      if len(possibleScales) == 0:
        scalesToCheck = np.linspace(0.2, 2.0, 80)[::-1]
      else:
        lastScale = possibleScales[-1][0]
        scaleOffset = 0.12 / len(possibleScales)
        scalesToCheck = np.concatenate([
          np.geomspace(lastScale, lastScale * (1 - scaleOffset), 20),
          np.geomspace(lastScale, lastScale * (1 + scaleOffset), 20)
        ])

      # Check all scales and templates.
      for scale in scalesToCheck:
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        for template in templateToCheck:
          if resized.shape[0] < template.shape[0] or resized.shape[1] < template.shape[1]:
            break
          result = cv2.matchTemplate(resized, template, cv2.TM_CCORR_NORMED)
          (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
          if maxVal >= self.state["matchThreshold"] and (scale not in scaleToMaxVal or scaleToMaxVal[scale] < maxVal):
            scaleToMaxVal[scale] = maxVal

      if len(scaleToMaxVal) > 0:
        scaleCandidate = max(scaleToMaxVal, key=scaleToMaxVal.get)
        possibleScales.append((scaleCandidate, scaleToMaxVal[scaleCandidate]))

    if len(possibleScales) < 7:
      raise Exception(f'Failed to calibrate scale from {self.videoFilePath}')
    lastScale = possibleScales[-1][0]
    self.state["scale"] = lastScale
  
  def getTemplateLocation(self, grayResized, template):
    (_, maxVal, _, location) = cv2.minMaxLoc(cv2.matchTemplate(grayResized, template, cv2.TM_CCORR_NORMED))
    if maxVal >= self.state["matchThreshold"]:
      return location
  
  def calibrateMapTextLocation(self):
    if all(self.state[x] is not None for x in ["scoreLoc", "turnLoc", "apLoc"]):
      return
    print('Calibrating map text location...')
    possibleScoreLoc = None
    possibleTurnLoc = None
    for frameIndex in range(0, self.totalFrames, math.floor(self.fps * 60)):
      if all(self.state[x] is not None for x in ["scoreLoc", "turnLoc", "apLoc"]):
        return
      print(f'  {secondsToTime(math.floor(frameIndex / self.fps))} / {self.totalDurationString}...')

      resized = self.getResizedFrame(frameIndex)
      if resized is None:
        continue
      grayResized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
      radarMaxLoc = self.getTemplateLocation(grayResized, self.radarTemplate)
      resources2MaxLoc = self.getTemplateLocation(grayResized, self.resources2Template)
      if radarMaxLoc is None or resources2MaxLoc is None:
        continue

      if self.state["scoreLoc"] is None:
        scoreLoc = getScoreLocation(
          self.radarTemplate, radarMaxLoc,
          self.resources2Template, resources2MaxLoc)
        scoreProcessed = processScoreImage(crop(grayResized, scoreLoc))
        score, scoreOcrString = self.gflOcr.recognize(scoreProcessed, ocr.OcrTargets.SCORE)
        if score is not None:
          self.state["scoreLoc"] = scoreLoc
        else:
          possibleScoreLoc = scoreLoc
      if self.state["turnLoc"] is None:
        turnLoc = getTurnLocation(
          self.radarTemplate, radarMaxLoc,
          self.resources2Template, resources2MaxLoc)
        turnProcessed = processTurnImage(crop(grayResized, turnLoc))
        turn, turnOcrString = self.gflOcr.recognize(turnProcessed, ocr.OcrTargets.TURN)
        if turn is not None:
          self.state["turnLoc"] = turnLoc
        else:
          possibleTurnLoc = turnLoc
      if self.state["apLoc"] is None:
        apLoc = getApLocation(
          self.radarTemplate, radarMaxLoc,
          self.resources2Template, resources2MaxLoc)
        apProcessed = processApImage(crop(resized, apLoc))
        ap, apOcrString = self.gflOcr.recognize(apProcessed, ocr.OcrTargets.AP_LEFT)
        if ap is not None:
          self.state["apLoc"] = apLoc
    if self.state["scoreLoc"] is None:
      print('  Warning: couldn\'t confirm score location... Is this a Sijun VOD?')
      self.state["scoreLoc"] = possibleScoreLoc
    if self.state["turnLoc"] is None:
      print('  Warning: couldn\'t confirm turn number location... Is this a Sijun VOD?')
      self.state["turnLoc"] = possibleTurnLoc
    if any(self.state[x] is None for x in ["scoreLoc", "turnLoc", "apLoc"]):
      raise Exception(f'Failed to calibrate map text location from {self.videoFilePath} with scale {self.state["scale"]}')

  def getFrameInfo(self, frameIndex):
    resized = self.getResizedFrame(frameIndex)
    if resized is None:
      return {
        "index": frameIndex,
        "state": VideoState.UNKNOWN
      }
    grayResized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    radarMaxLoc = self.getTemplateLocation(grayResized, self.radarTemplate)
    resources2MaxLoc = None
    if radarMaxLoc is None:
      resources2MaxLoc = self.getTemplateLocation(grayResized, self.resources2Template)
    if radarMaxLoc is not None or resources2MaxLoc is not None:
      scoreCropped = crop(grayResized, self.state["scoreLoc"])
      scoreProcessed = processScoreImage(scoreCropped)
      turnProcessed = processTurnImage(crop(grayResized, self.state["turnLoc"]))
      apProcessed = processApImage(crop(resized, self.state["apLoc"]))
      info = {
        "index": frameIndex,
        "state": VideoState.MAP,
        #"score": (score, scoreOcrString),
        "score": self.gflOcr.recognize(scoreProcessed, ocr.OcrTargets.SCORE),
        "turn": self.gflOcr.recognize(turnProcessed, ocr.OcrTargets.TURN),
        "ap": self.gflOcr.recognize(apProcessed, ocr.OcrTargets.AP_LEFT)
      }
      #cv2.imwrite(self.videoFilePath + f'__debug/{frameIndex}_scoreCropped__{info["score"][1]}.png', scoreCropped)
      #cv2.imwrite(self.videoFilePath + f'__debug/{frameIndex}_scoreProcessed__{info["score"][1]}.png', scoreProcessed)
      #if info["turn"][0] is None:
      #cv2.imwrite(self.videoFilePath + f'__debug/{frameIndex}_turnProcessed__{info["turn"][1]}.png', turnProcessed)
      #if info["ap"][0] is None:
      #  cv2.imwrite(self.videoFilePath + f'__debug/{frameIndex}_apProcessed__{info["ap"][1]}.png', apProcessed)
      return info

    battlePauseMaxLoc = self.getTemplateLocation(grayResized, self.battlePauseTemplate)
    if battlePauseMaxLoc is not None:
      return {
        "index": frameIndex,
        "state": VideoState.COMBAT
      }

    loadingLoc = self.getTemplateLocation(grayResized, self.loadingTemplate)
    if loadingLoc is not None:
      return {
        "index": frameIndex,
        "state": VideoState.LOADING
      }

    return {
      "index": frameIndex,
      "state": VideoState.UNKNOWN
    }
  
  def parseCompleteFrameInfo(self):
    def shouldProcessIntermediateFrames(prevInfo, curInfo):
      intervalSeconds = math.floor((curInfo["index"] - prevInfo["index"]) / self.fps)
      # It's pretty unlikely for the game to go from one battle to another within 2s.
      if intervalSeconds <= 2 and (
           (prevInfo["state"] == VideoState.COMBAT and curInfo["state"] == VideoState.LOADING) or
           (prevInfo["state"] == VideoState.LOADING and curInfo["state"] == VideoState.COMBAT) or
           (prevInfo["state"] == VideoState.LOADING and curInfo["state"] == VideoState.LOADING)
         ):
        return False
      if prevInfo["state"] != curInfo["state"]:
        return True
      if prevInfo["state"] == VideoState.MAP:
        if curInfo["ap"][0] is not None and curInfo["ap"][0] == prevInfo["ap"][0]:
          return False
        if curInfo["score"][0] is None or prevInfo["score"][0] is None:
          return True
        if curInfo["score"][0] != prevInfo["score"][0]:
          return True
      # The two clauses below are for catching quick map movement, such as planning mode.
      elif prevInfo["state"] == VideoState.COMBAT and intervalSeconds >= 8:
        return True
      elif prevInfo["state"] == VideoState.LOADING and intervalSeconds >= 4:
        return True
      return False

    if "curGranularity" not in self.state:
      self.state["startingGranularity"] = STARTING_GRANULARITY
      self.state["curGranularity"] = STARTING_GRANULARITY
      self.state["frameInfos"] = []

    totalPasses = int(math.log2(self.state["startingGranularity"] / MINIMUM_GRANULARITY)) + 1
    checkpointFrameCount = math.floor(self.fps * 60 * 10)
    existingIndices = set([x["index"] for x in self.state["frameInfos"]])
    while self.state["curGranularity"] > 0:
      curPass = int(math.log2(self.state["startingGranularity"] / self.state["curGranularity"])) + 1
      curGranularityFrames = math.floor(self.state["curGranularity"] * self.fps)

      framesToProcess = []
      if self.state["curGranularity"] == self.state["startingGranularity"]:
        framesToProcess = list(range(0, self.totalFrames, curGranularityFrames))
        if framesToProcess[-1] < self.totalFrames - 1:
          framesToProcess.append(self.totalFrames - 1)
      else:
        prevInfo = self.state["frameInfos"][0]
        for curInfo in self.state["frameInfos"][1:]:
          if shouldProcessIntermediateFrames(prevInfo, curInfo):
            framesToProcess.append(int((prevInfo["index"] + curInfo["index"]) / 2))
          prevInfo = curInfo
      totalFramesToProcess = len(framesToProcess)
      totalFramesToProcessStrlen = len(str(totalFramesToProcess))

      print(f'Parsing video - pass {curPass} / {totalPasses} - ' + 
            f'{self.state["curGranularity"]}s steps, ' +
            f'{totalFramesToProcess} frames to process')

      lastCheckpointFrame = 0
      for index, frameIndex in enumerate(framesToProcess):
        if frameIndex >= lastCheckpointFrame + checkpointFrameCount:
          checkpointString = secondsToTime(math.floor(frameIndex / self.fps))
          indexStr = str(index).rjust(totalFramesToProcessStrlen, " ")
          percentage = str(math.floor(index / totalFramesToProcess * 100)).rjust(2)
          print(f'  {percentage}% = {indexStr} / {totalFramesToProcess} frames ' +
                f'@ {checkpointString} / {self.totalDurationString}...')
          lastCheckpointFrame = frameIndex
          self.save()
        #print(f'    {frameToDebugString(self.fps, frameIndex)}')
        if frameIndex in existingIndices:
          continue
        self.state["frameInfos"].append(self.getFrameInfo(frameIndex))
        existingIndices.add(frameIndex)
      self.state["frameInfos"].sort(key=lambda x: x["index"])
      
      if len(self.state["frameInfos"]) == 0:
        raise Exception("No frames?")
   
      nextGranularity = self.state["curGranularity"] / 2
      if nextGranularity < MINIMUM_GRANULARITY:
        self.state["curGranularity"] = 0
        self.save()
        break

      self.state["curGranularity"] = nextGranularity
      self.save()

  def produceSegmentGroups(self):
    if self.segmentGroups is None:
      segments = []
      segmentStartFrame = None
      firstValidFrame = None
      previousValidFrame = None
      for frameInfo in self.state["frameInfos"]:
        if frameInfo["state"] == VideoState.UNKNOWN:# or frameInfo["state"] == VideoState.LOADING:
          continue
        if previousValidFrame is not None and previousValidFrame["state"] != frameInfo["state"]:
          segments.append((firstValidFrame, previousValidFrame))
          firstValidFrame = None
          previousValidFrame = None
        if frameInfo["state"] == VideoState.COMBAT or frameInfo["state"] == VideoState.LOADING or frameInfo["score"][0] is not None:
          previousValidFrame = frameInfo
          if firstValidFrame is None:
            firstValidFrame = frameInfo
      if firstValidFrame is not None:
        segments.append((firstValidFrame, previousValidFrame))

      segmentGroups = []
      curSegments = []
      loadingSegments = []
      for segment in segments:
        if segment[0]["state"] == VideoState.LOADING:
          if len(curSegments) > 0:
            loadingSegments.append(segment)
        # Flush group
        elif len(curSegments) > 0 and curSegments[0][0]["state"] != segment[0]["state"]:
          segmentGroups.append((curSegments, loadingSegments))
          curSegments = [segment]
          loadingSegments = []
        else:
          curSegments.append(segment)
          
      self.segmentGroups = segmentGroups

      totalSec = dict([(x, 0) for x in [VideoState.MAP, VideoState.COMBAT, VideoState.LOADING]])
      totalLoadingSec = dict([(x, 0) for x in [VideoState.MAP, VideoState.COMBAT]])
      for segments, loadingSegments in segmentGroups:
        for start, end in segments + loadingSegments:
          startSec = math.floor(start["index"] / self.fps)
          endSec = math.floor(end["index"] / self.fps)
          totalSec[start["state"]] += endSec - startSec
          if start["state"] == VideoState.LOADING:
            totalSec[segments[0][0]["state"]] += endSec - startSec
      print('\n'.join([
        "Stats for nerds:",
      ] + [
        f'  Time spent {description}: {totalSec[state]}s = {secondsToTime(totalSec[state])}'
        for state, description in [
          (VideoState.MAP, "in map (w/ loading/reset)"),
          (VideoState.COMBAT, "in combat (w/ loading/reset)"),
          (VideoState.LOADING, "purely in loading/reset screens"),
        ]
      ]))
    return self.segmentGroups

  def writeSegmentsTsv(self):
    segmentGroups = self.produceSegmentGroups()
    with open(self.videoFilePath + '_segments.tsv', 'w') as f:
      header = '\t'.join(["Turn Start", "AP Start", "Turn End", "AP End", "Score Start", "Score End",
                          "Start", "End", "Duration (s)", "State", "# of Loading/Reset"])
      #print(header, flush=True)
      f.write(header + '\n')
      for segments, loadingSegments in segmentGroups:
        start = segments[0][0]
        end = segments[-1][1]
        startSec = math.floor(start["index"] / self.fps)
        endSec = math.floor(end["index"] / self.fps)
        columns = ["", "", "", "", "", ""]
        if start["state"] == VideoState.MAP:
          startScore = start["score"][0]
          endScore = end["score"][0]
          columns = [
            start["turn"][0],
            start["ap"][0],
            end["turn"][0],
            end["ap"][0],
            startScore,
            endScore
          ]
        columns.extend([
          secondsToTime(startSec),
          secondsToTime(endSec),
          endSec - startSec,
          start["state"]
        ])
        if start["state"] == VideoState.COMBAT:
          columns.extend([len(loadingSegments) - 1 if len(loadingSegments) > 0 else 0])
        else:
          columns.extend([""])
          
        line = '\t'.join(str(x) for x in columns)
        #print(line, flush=True)
        f.write(line + '\n')

  def writeMapMovementCompilationScript(self):
    # Bad things are likely to happen if the filename has ' or " in it...
    if re.search(r'[\'"]', self.videoFilePath):
      raise Exception("Filename cannot have an apostrophe or quotation mark.")

    segmentGroups = self.produceSegmentGroups()
    def isSegmentGroupEligible(segmentGroup):
      start = segmentGroup[0][0][0]
      end = segmentGroup[0][-1][1]
      return start["state"] == VideoState.MAP and start["index"] < end["index"] - self.fps
    eligibleSegmentGroups = list(filter(isSegmentGroupEligible, segmentGroups))

    videoFileBasename = os.path.basename(self.videoFilePath)
    outputFilePath = '"${SCRIPT_DIR}"/' + f'\'{videoFileBasename}_map_only.mkv\''
    clipFilePaths = []
    quietSettings = '-y -hide_banner -loglevel error -nostdin'
    lines = [
      '#!/bin/sh',
      'if ! type "ffmpeg" > /dev/null ; then echo "No ffmpeg found! Exiting..."; exit 1; fi',
      'if ! type "mkfifo" > /dev/null ; then echo "No mkfifo found! Exiting..."; exit 1; fi',
      'SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)',
      f'if [ -f {outputFilePath} ] ; then',
      '  echo -n "Existing map-only compilation found. Press y<Enter> to overwrite: "',
      '  read x',
      '  [ "$x" == "y" ] || (echo "Not overwriting"; exit 2)',
      'fi',
      'echo "Making map-only compilation video..."',
      'echo "Please note that this automated script is imperfect."',
      'date=$(date +%s)',
      'echo > /tmp/gfl_${date}_list.txt'
    ]
    pipeNames = [f'/tmp/gfl_${{date}}_{segments[0][0]["index"]}' for segments, _ in eligibleSegmentGroups]
    lines.extend([f'mkfifo {x}' for x in pipeNames])
    lines.extend([f'echo "file \'{x}\'" >> /tmp/gfl_${{date}}_list.txt' for x in pipeNames])
    for segments, _ in eligibleSegmentGroups:
      start = segments[0][0]
      end = segments[-1][1]
      startSec = math.floor(start["index"] / self.fps)
      endSec = math.floor(end["index"] / self.fps)
      fastSeekStartSec = startSec
      if fastSeekStartSec < 0:
        fastSeekStartSec = 0
      offsetEndSec = endSec - fastSeekStartSec
      lines.append(' '.join([
        f'ffmpeg {quietSettings}',
        f'-ss {secondsToTime(fastSeekStartSec)}',
        f'-i \'{self.videoFilePath}\'',
        '-f matroska -c copy',
        f'-to {secondsToTime(offsetEndSec)}',
        f'/tmp/gfl_${{date}}_{start["index"]}',
        '&'
      ]))
    lines.append(f'ffmpeg {quietSettings} -f concat -safe 0 -i /tmp/gfl_${{date}}_list.txt -c copy {outputFilePath}')
    lines.append('echo "Cleaning up temporary files..."')
    lines.append('rm /tmp/gfl_${date}_list.txt 2>/dev/null')
    lines.extend(['rm ' + x + ' 2>/dev/null' for x in pipeNames])
    lines.append('')
    with open(self.videoFilePath + '_compileMapOnlyVideo.sh', 'w') as f:
      f.write('\n'.join(lines))

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print(f'python {sys.argv[0]} {file}')
    sys.exit(1)
  if not os.path.exists(sys.argv[1]):
    print(f'{sys.argv[1]} doesn\'t exist')
    sys.exit(1)
  scriptDir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
  parser = VideoParser(scriptDir, sys.argv[1])
  parser.makeTemplates()
  parser.calibrateScaling()
  parser.calibrateMapTextLocation()
  parser.save()
  parser.parseCompleteFrameInfo()
  parser.writeSegmentsTsv()
  parser.writeMapMovementCompilationScript()

