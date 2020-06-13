## yolov2对于iou的计算

- [参考](https://blog.csdn.net/qq_17550379/article/details/78815637)

- 计算输入值
  - x:表示box中心x坐标
  - y:表示box中心y坐标
  - w：表示box的宽度
  - h：表示box的高度

- 在一张img中，左上角的坐标是(0，0),往右跟往下坐标递增

- demo1

```python
 '''
  这里boxes1,2最后一个纬度存放的是 [x, y , sqrt(w), sqrt(h)]
 '''
    def calc_iou(self, boxes1, boxes2):
        #提取w,h。因为yolov2中是用Sqrt(w) 和 sqrt(h)计算的，所以乘了平方
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        
        # w*h
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        
        # x - w*0.5 这是box1的左边的中心
        # y - w*0.5 上边中心
        # x + w*0.5 右边中心
        # y + w*0.5 下边中心
        box = tf.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))
		
        # 相交面积左边中心，上边中心
        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        
        # 相交面积右边中心，下边中心
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])
		
        # 右-左，下-上
        intersection = tf.maximum(right_down - left_up, 0.0)
        
        # w*h即面积
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        	
        # 并    
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)

```



- demo2

  ```python
  float overlap(float x1, float w1, float x2, float w2)
  {
      float l1 = x1 - w1/2;
      float l2 = x2 - w2/2;
      float left = l1 > l2 ? l1 : l2;
      float r1 = x1 + w1/2;
      float r2 = x2 + w2/2;
      float right = r1 < r2 ? r1 : r2;
      return right - left;
  }
  
  float box_intersection(box a, box b)
  {
      float w = overlap(a.x, a.w, b.x, b.w);
      float h = overlap(a.y, a.h, b.y, b.h);
      if(w < 0 || h < 0) return 0;
      float area = w*h;
      return area;
  }
  
  float box_union(box a, box b)
  {
      float i = box_intersection(a, b);
      float u = a.w*a.h + b.w*b.h - i;
      return u;
  }
  float box_iou(box a, box b)
  {
      return box_intersection(a, b)/box_union(a, b);
  }
  
  ```

  