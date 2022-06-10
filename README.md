# Crosswalk detection with Yolov5 and Deepsort

Yolov5 and Deepsort parts from: [Yolov5_DeepSort_OSNet repo](https://github.com/mikel-brostrom/Yolov5_DeepSort_OSNet).

## Before runing the tracker

Install requirements:

`pip install -r requirements.txt`

## Run tracker

```bash
python3 crosswalk-detect.py --input 0  # webcam
                                    img.jpg  # image
                                    vid.mp4  # video
                            --show-vid # show the video during processing
                            --save-vid # save video to file
                            --output-dir # the output directory of the video
```
