# ObamaNet : Photo-realistic lip-sync from text (Unofficial port)

### 1. Data
---
I'm scraping data from youtube using youtube-dl.
Currently working on the subtitle extraction using command like the following from the cli.
```
youtube-dl --sub-lang en --skip-download --write-sub --output '~/obamanet/data/captions/%(autonumber)s.%(ext)s' --batch-file ~/obamanet/data/obama_addresses.txt --ignore-config

```
