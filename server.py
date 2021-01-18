from flask import Flask, render_template, request, redirect
from flask import send_from_directory, url_for
from werkzeug.utils import secure_filename
import numpy as np
import os
import glob
import cv2
import config as cfg
import pickle
from selective_search import selective_search
from keras.models import load_model
from keras.models import Model
from sklearn.metrics import average_precision_score
from nms import soft_nms, hard_nms
from iou import iou
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png']
app.config['UPLOAD_PATH'] = './uploads'

extractor = None
classifier = None

# Get feature extractor model and classify model
def get_model():
	global extractor
	global classifier

	# Load feature extractor model
	model = load_model(cfg.MODEL_PATH)
	output = model.get_layer(name='activation_5').output
	extractor = Model(inputs=model.input, outputs=output)

	# Load classifier model
	f = open(cfg.CLASSIFY_PATH, 'rb')
	classifier = pickle.load(f)
	f.close()

# Predict
def predict(filename):
	# Open file to write result
	f = open(os.path.join(app.config['UPLOAD_PATH'], 'result.txt'), 'w')

	# Load image
	image = cv2.imread(os.path.join(app.config['UPLOAD_PATH'], filename))

	# Get image shape
	ih,iw = image.shape[:2]

	# Selective search
	start_time = time.time()
	rects = selective_search(image, method='quality', verbose=False, display=False)
	f.write('* Selective search: {} seconds *\n'.format(time.time() - start_time))

	# Initialize bounding box
	start_time = time.time()
	rp = []
	for (x,y,w,h) in rects[:2000]:
		# Get predicted
		window = image[y:y+h, x:x+w]
		# Resize to target size
		im_rsz = cv2.resize(window, cfg.IMG_SIZE, interpolation=cv2.INTER_AREA)
		# Append to list
		rp.append(im_rsz)

	# Normalization
	rp = np.array(rp, dtype=np.float)/255.0

	# Extract features
	features = extractor.predict(rp)
	f.write('* Feature extraction: {} seconds *\n'.format(time.time() - start_time))

	# Classify
	start_time = time.time()
	pred = classifier.predict_proba(features)
	f.write('* Classification: {} seconds *\n'.format(time.time() - start_time))	
	
	# Find every bounding box with probability greater than threshold
	index = np.where(np.max(pred[:,1:], axis=1) >= 0.9)[0]
	pred = pred[index,:]
	bbox = rects[index,:]
	# print('[INFO] Found {} bounding box'.format(len(bbox)))

	# Non-Maximal Suppression for each class
	labels = np.argmax(pred, axis=1)
	uniqueLabel = np.unique(labels)
	result = np.array([])

	for label in uniqueLabel:
		labelIdx = np.where(labels == label)[0]
		boxes = bbox[labelIdx,:]
		scores = pred[labelIdx,label]

		# if args['method'] == 'soft':
		nms_boxes, nms_scores = soft_nms(boxes, scores, threshold=0.9)
		nms_scores = nms_scores[:,np.newaxis]
		# elif args['method'] == 'hard':
		# 	nms_boxes, nms_scores = hard_nms(boxes, scores, overlapThresh=args['threshold'])
		# 	nms_scores = nms_scores[:,np.newaxis]
		# else:
		# 	nms_boxes, nms_scores = boxes, scores
		# 	nms_scores = nms_scores[:,np.newaxis]

		label_box = np.hstack((nms_boxes, nms_scores, np.full((len(nms_scores),1), label)))
		if len(result):
			result = np.vstack((result, label_box))
		else:
			result = label_box
	# print('[INFO] After NMS there are {} bounding box'.format(len(result)))

	# Write result
	for x,y,w,h,prob,label in result:
		f.write('- Label {}: score {}\n'.format(label, np.around(prob,3)))
		
		red = (0,0,255)
		green = (0,255,0)
		blue = (255,0,0)
		black = (0,0,0)

		cv2.rectangle(image, (int(x),int(y)), (int(x+w),int(y+h)), color=green, thickness=1)
		cv2.putText(image, text='{}:{}'.format(int(label), np.around(prob,2)), org=(int(x)+5,int(y)+5), 
			fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, color=red)
	cv2.imwrite(os.path.join(app.config['UPLOAD_PATH'],filename),image)
	f.close()


@app.route('/hello/')
def hello():
	return 'Hello world'

@app.route('/')
def index():
	try:
		# Display only the recently uploaded image
		print('[DBG] Branch try')
		filename = max(glob.glob(app.config['UPLOAD_PATH'] + '/*.png'), key=os.path.getctime)
		filename = [filename.split('\\')[-1]]
		print('[DBG] filename: ', filename)
		with open(os.path.join(app.config['UPLOAD_PATH'],'result.txt')) as f:
			content = f.read()
		return render_template('index.html',files=filename,content=content)
	except:
		print('[DBG] Branch except')
		return render_template('index.html')

@app.route('/uploads/<filename>')
def upload(filename):
	return send_from_directory(app.config['UPLOAD_PATH'], filename)

@app.route('/', methods=['POST'])
def upload_file():
	uploaded_file = request.files['file']
	filename = secure_filename(uploaded_file.filename)
	if filename != '':
		file_ext = os.path.splitext(filename)[1]
		if file_ext not in app.config['UPLOAD_EXTENSIONS']:
			abort(400)
		# filename = 'image' + file_ext
		uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'],filename))
		predict(filename)
	return redirect(url_for('index'))

if __name__ == '__main__':
	get_model()
	app.run()	