import mxnet as mx
import gluoncv as gcv
import matplotlib.pyplot as plt
from mxnet.gluon.data.vision import transforms
from gluoncv.data.ade20k.segmentation import ADE20KSegmentation
from gluoncv.utils.viz import get_color_pallete

img_filepath = "stock/tennisball.jpg"

image = mx.image.imread(img_filepath)

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])

image = transform_fn(image)

image = image.expand_dims(0)

network = gcv.model_zoo.get_fcn_resnet50_ade(pretrained=True)

output = network.demo(image)

output = output[0]
# number of classes, img height, img width
print(output.shape)

px_height, px_width = 300, 500
px_logit = output[:, px_height, px_width]

px_prob = mx.nd.softmax(px_logit)
px_rounded_prob = mx.nd.round(px_prob*100)/100
print(px_rounded_prob)

class_index = mx.nd.argmax(px_logit, axis=0)
class_index = class_index[0].astype('int').asscalar()
print(class_index)

class_label = ADE20KSegmentation.CLASSES[class_index]
print(class_label)

# # But by specifying axis equals 0, we can apply softmax independently for all pixels. Axis zero corresponds to the channel dimension which are the classes.
# output_prob = mx.nd.softmax(output, axis=0)

# output_heatmap = output_prob[127]
# # plt.imshow(output_heatmap.asnumpy())
# # plt.show()

prediction = mx.nd.argmax(output, 0).asnumpy()
# print(prediction.shape)

predictionimage = get_color_pallete(prediction, 'ade20k')

plt.imshow(predictionimage)
plt.show()
