## 简单说明
根据码盘角度，将距离当前帧180度左右的两圈点云作为与其匹配的局部地图，使用面特征点。
![](./pic/局部地图示意图.png)

## 运行方式
```
source devel/setup.bash
roslaunch le_calib calib.launch
```