## How to use 
Protect your video even if it's masked or labeled.

![Video Detection Example](https://raw.githubusercontent.com/keo-dara/detect_similar_video/refs/heads/main/image.png)

### New Environment
```
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```
pip install -r requirements.txt
```


### Generate dataset
```
python video_title_annotate.py
```

2. Train model
```
python train.py
```

3. Detect image
```
python detect_image.py
```

4. Detect video
```
python detect_video.py
```
