```bash
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

## Results:
### cascade detector
```text
left + right ear: Average IOU: 30.35%
```

### v2
Resized to 416x416
```text
v2 (round): Average IOU: 87.74%
v2 (floor): Average IOU: 85.83%
v2 (ceil) : Average IOU: 87.49%
```

### v3
Resized to 416x416, blur
```text
v3 (round): Average IOU: 85.53%
v3 (floor): Average IOU: 84.16%
v3 (ceil): Average IOU: 84.82%
```