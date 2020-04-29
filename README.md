# VFX_Stitching
Homework #2 in NTU Digital Visual Effects, Spring 2020.

By 蘇俐文、李其蓉
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
## Implementation
Feature detection 的部分實作了 Harris Corner Detection。
參數滿難調的，因為照片裡一樣明顯的feature points數量不同，如果條件太鬆就會找到太多不必要的點，讓matching出現錯誤或是很慢；條件太嚴格就會找不到足夠的對應點，做不好matching。

Feature matching 則實作了spatial neighbors comparison，一開始只用自己+相鄰的3x3個pixel，發現找不太到matching features，後來**加大搜尋的範圍**，希望增加可容許誤差值，改成找5x5的pixels後就找到比較多matching point。

接著對每張圖和feature points座標做 Cylindrical Warping，一開始忘記做**feature points的座標轉換**，所以結果一直不好XD

轉換完後用RANSAC找出最合適的 translation model，一開始按照公式決定做 K 次 sample，但可能是找到的matching points不夠多，或是有對應的點比例太小，也有遇到一直取到一樣的sample導致出現很差結果的狀況，且實際上找到的對應點的數量不多，就改成直接**暴力搜尋**每個可能，就得到比較好的結果了。

Stitching的部分，實作兩種blending方法，分別是簡單的alpha blending和在差異最小的地方做小範圍alpha blending(我們稱作min error alpha blending)。
alpha blending在場景中有移動物體時就會產生鬼影，但在**室內**的效果還不錯。

而如果使用min error alpha blending可以消除鬼影(高頻)，但在低頻區域(例如一大片草地)就會出現有點明顯的邊界。由於時間關係我們沒有**對不同頻率的區域做不同blending**，如果有做到，我們覺得能得到更好的結果。

最後可以選擇輸出完整為剪裁的圖片或是對圖片做裁剪，關於剪裁方法我們實作了兩種:Direct clip和Rotate clip。Direct clip直接對漂移的影像作長方形裁切，這樣會剪掉很多上下有影像的部分，所以我們想了**Rotate clip**方法，把stitch後的影像按照偏移的量轉回比較正的長方形，並對長方形做裁切，這樣可以保留比較多上下部分的影像(原本400px變成500px)。

## Result
右鍵看大圖
### [parrington](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/assignments/proj2/data/parrington.zip)
images from VFX course website
#### Simple alpha blending
![](https://i.imgur.com/e4k2nvb.jpg)
#### Min-error alpha blending
![](https://i.imgur.com/DoFStqY.jpg)
#### Clip
![](https://i.imgur.com/0tMZNhy.jpg)
#### Rotate clip
![](https://i.imgur.com/htgcnQb.jpg)


### [grail](https://www.csie.ntu.edu.tw/~cyy/courses/vfx/20spring/assignments/proj2/data/grail.zip)
images from VFX course website
#### Simple alpha blending
![](https://i.imgur.com/q63bbfF.jpg)
#### Min-error alpha blending
![](https://i.imgur.com/RbnWIMN.jpg)
#### Clip
![](https://i.imgur.com/iv1n0Ct.jpg)
#### Rotate clip
![](https://i.imgur.com/wT6sAIh.jpg)

