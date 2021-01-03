import argparse
import random
import time
import cv2

def selective_search(image, method='fast', verbose=True, display=False):
	"""
	Perform selective search to find region proposals.
	@INPUT:
		- image: image need to find region proposals
		- method: selective search method. method=['fast', 'quality'], default = 'fast'
	@OUTPUT:
		- rects: Bounding box contain coordinate x, y and size w,h
	"""
	# load the input image
	# image = cv2.imread(args['image'])

	# initialize OpenCV selective search implementation and set the input image
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)

	# check to see if we are using the 'fast' or 'quality' method
	if method == 'fast':
		ss.switchToSelectiveSearchFast()
	else:
		ss.switchToSelectiveSearchQuality()

	# Run selective search on the input image
	try:
		start = time.time()
		rects = ss.process()
		end = time.time()
	except:
		print('[ERROR] Can\'t process. Skip this image')
		return []

	# show how long selective search took to run along with the total number of
	# returned region proposals
	if verbose:
		print('[INFO] using *{}* selective search'.format(method))
		print('[INFO] selective search took {:.4f} seconds'.format(end - start))
		print('[INFO] {} total region proposals'.format(len(rects)))

	# Display image
	if display:
		for i in range(0, len(rects), 100):
			output = image.copy()

			for (x, y, w, h) in rects[i:i+100]:
				color = [random.randint(0,255) for j in range(0,3)]
				cv2.rectangle(output, (x,y), (x+w,y+h), color, 2)

			cv2.imshow('Output', output)
			key = cv2.waitKey(0) & 0xff

			if key == ord('q'):
				break

	return rects

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument('-i', '--image', required=True, help='path to the input image')
	ap.add_argument('-m', '--method', type=str, default='fast', 
		choices=['fast', 'quality'], help='selective search method')
	args = vars(ap.parse_args())

	image = cv2.imread(args['image'])
	method = args['method']

	selective_search(image, method, verbose=True, display=True)