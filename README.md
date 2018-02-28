# ObamaNet : Photo-realistic lip-sync from text (Unofficial port)

### 0. Requirements
---
* Python 3.5+
* ffmpeg (`sudo apt-get install ffmpeg`)
* [YouTube-dl](https://github.com/rg3/youtube-dl#video-selection)

### 1. Data Extraction
---
I extracted the data from youtube using youtube-dl. It's perhaps the best downloader for youtube on linux. Commands for extracting particular streams are given below.

* Subtitle Extraction
```
youtube-dl --sub-lang en --skip-download --write-sub --output '~/obamanet/data/captions/%(autonumber)s.%(ext)s' --batch-file ~/obamanet/data/obama_addresses.txt --ignore-config
```
* Video Extraction
```
```
* Video to Audio Conversion
```
```
