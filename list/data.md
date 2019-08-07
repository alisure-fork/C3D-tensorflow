## Data


1. 安装[ffmpeg](http://www.ffmpeg.org/download.html)
```
sudo apt-get install yasm
tar -jxvf ffmpeg-x.x.tar.bz2
cd ffmpeg-x.x
./configure
make
make install
```


2. 安装jot命令

```
sudo apt-get install athena-jot
```


3. 下载 [UCF101](http://crcv.ucf.edu/data/UCF101.php) (Action Recognition Data Set)


4. Each single avi file is decoded with 5FPS (it's depend your decision) in a single directory
```
./list/convert_video_to_images.sh /home/ubuntu/data1.5TB/C3D/UCF-101 5
```


5. 生成`train.list`和`test.list`文件(每一行包括：图片目录 类别) 
 
```
./list/convert_images_to_list.py
```
