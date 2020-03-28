# organize imports
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils, MobileNetV2

# process an image to be mobilenet friendly
def process_image(img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	img_array = image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)
	pImg = mobilenet.preprocess_input(img_array)
	return pImg

# main function
if __name__ == '__main__':

	# path to test image
	test_img_path = "/content/drive/My Drive/AI - build/train2014/images/train2014/COCO_train2014_000000581637.jpg"

	# process the test image
	pImg = process_image(test_img_path)

	# define the mobilenet model
	model = MobileNetV2(weights = 'imagenet')
	# make predictions on test image using mobilenet
	prediction = model.predict(pImg)

	# obtain the top-5 predictions
	results = imagenet_utils.decode_predictions(prediction)
	print(results)