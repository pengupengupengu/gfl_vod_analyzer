# Girls' Frontline VOD Analyzer

Are you one out of maybe 500 people worldwide who care about Girls' Frontline ranking? Do you spend hours watching other rankers' videos?

Well, do I have a program for you! Now you can feed this program hours of GFL ranking footage and it will spit out a nice TSV table of scores and times so you can easily seek around the VOD.

![](https://i.imgur.com/B2BcGLp.png)

Note: the Enemy CE column is produced in Sheets by multiplying at the score difference before and after a combat section by 10.

## Prereqs

You need Python 3 and a pretty typical list of packages:

```
pip install opencv-python imutils numpy
```

## Usage

Just point it at a VOD:

```
python video_match2.py vod.mkv
```

On a Ryzen 5600 it takes about 10-15 minutes per 4 hours of footage. After it completes, it will output a `vod.mkv_segments.tsv` TSV file and a `vod.mkv__compileMapOnlyVideo.sh`.

If you are using a POSIX system (though WSL and Cygwin should work?) and have `ffmpeg` installed, then you can run `vod.mkv__compileMapOnlyVideo.sh` to create a cut version of `vod.mkv` that only contains map movements.

```
chmod +x vod.mkv__compileMapOnlyVideo.sh
./vod.mkv__compileMapOnlyVideo.sh
```

The compilation will be saved to `vod.mkv_map_only.mkv`. The compilation video is imperfect and can stutter a little bit.

## Design

The script scans every 16 seconds of the video, determines the state of the gameplay (map, combat, or loading screen), and tries to parse the current score, turn number, and AP left from map frames. Then it decides which intermediate frames to inspect. For example, if the current score didn't change between two map frames, then it will NOT inspect the intermediate frame. This algorithm will recurse with 8s, 4s, 2s, and 1s steps.

This algorithm isn't perfect, and I'm pretty new to Python and computer vision in general, so feel free to improve on it.

### Why kNN for OCRing score/turn/AP left?

Most OCR models are just not optimized for the score/turn number (Mohave) and AP left (Novecentowide Medium) fonts. I tried training models for EasyOCR, keras-ocr, Tesseract, etc. but in the end just throwing examples into kNN was faster and more accurate.

## Unimplemented Features

* CUDA-based processing: Somewhat likely
* Parsing dolls from battles: This is possible because we could extract the doll portraits from the bottom skill buttons bar, but I don't see much of a point
* Parsing enemies from battles: Unlikely
