import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt

img_filepath = "stock\imagewithstuff.jpg"

image = mx.image.imread(img_filepath)

# plt.imshow(image.asnumpy())
# plt.show()

image, chw_image = gcv.data.transforms.presets.yolo.transform_test(
    image, short=512)

# plt.imshow(chw_image)
# plt.show()
network = gcv.model_zoo.yolo3_darknet53_coco(pretrained=True)
prediction = network(image)

# number of image, number of possible classes, number of values to define bounding box
for index, array in enumerate(prediction):
    print('#', index+1, array.shape)

prediction = [array[0] for array in prediction]

class_indicies, prob, bound_box = prediction

k = 10
# print(class_indicies[:k])
# #16 is dog
# print(prob[:k])
# print(bound_box[:k])

gcv.utils.viz.plot_bbox(chw_image, bound_box, prob,
                        class_indicies, class_names=network.classes)
plt.imshow(chw_image)
plt.show()
