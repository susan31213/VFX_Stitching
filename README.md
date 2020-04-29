# VFX_Stitching

## Dependencies
+ Python 3.6.4
+ opencv-python 4.1.1
+ matplotlib
+ skimage

## Run
```
python hw2.py [-h] -n PREFIX_FILENAME -f FORMAT -i NUMBER -o OUTPUT
              [-b BLENDER] [-w BANDWIDTH] [-D] [-C]
              


A Python implementation of image stitching

optional arguments:
  -h, --help            show this help message and exit
  -n PREFIX_FILENAME, --prefix-filename PREFIX_FILENAME
                        image file path and image prefix name, need pano.txt
                        for example: parrington/prtn
  -f FORMAT, --format FORMAT
                        input image format
  -i NUMBER, --number NUMBER
                        the number of input images
  -o OUTPUT, --output OUTPUT
                        ouput file name
  -b BLENDER, --blender BLENDER
                        blend method when stitching, default is alpha [alpha,
                        min-error-alpha]
  -w BANDWIDTH, --bandwidth BANDWIDTH
                        bandwidth of min error alpha blend method, default is 3
  -D, --debug           option to show debug messages
  -C, --clip            option to clip into rect image without black boundary
```
## 


## Result
可右鍵>在新分頁開啟圖片看大圖
### [parrington](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/assignments/proj2/data/parrington.zip)
images from VFX course website
#### Simple alpha blending
![](https://i.imgur.com/e4k2nvb.jpg)
#### Min-error alpha blending
![](https://i.imgur.com/DoFStqY.jpg)
#### Clip
![](https://i.imgur.com/0tMZNhy.jpg)


### [grail](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/assignments/proj2/data/grail.zip)
images from VFX course website
#### Simple alpha blending
![](https://i.imgur.com/q63bbfF.jpg)
#### Min-error alpha blending
![](https://i.imgur.com/RbnWIMN.jpg)
#### Clip
![](https://i.imgur.com/iv1n0Ct.jpg)
