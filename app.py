from flask import Flask, request, Response
from flask.json import jsonify
import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import sys
import json
from operator import itemgetter
from flask_cors import CORS #pip install -U flask-cors
import base64
from serve import get_model_api  # see part 1.
import base64

IMG_SIZE = 28

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default
model_api = get_model_api()

all_chars = ['A', 'E/I', 'O/U', 'BA', 'BE/BI', 'BO/BU', 'B', 'KA', 'KE/KI', 'KO/KU', 'K', 'DA', 'DE/DI', 'DO/DU', 'D', 'GA', 'GE/GI', 'GO/GU', 'G', 'HA', 'HE/HI', 'HO/HU', 'H', 'LA', 'LE/LI', 'LO/LU', 'L', 'MA', 'ME/MI', 'MO/MU', 'M', 'NA', 'NE/NI', 'NO/NU', 'N', 'NGA', 'NGE/NGI', 'NGO/NGU', 'NG', 'PA', 'PE/PI', 'PO/PU', 'P', 'SA', 'SE/SI', 'SO/SU', 'S', 'TA', 'TE/TI', 'TO/TU', 'T', 'WA', 'WE/WI', 'WO/WU', 'W', 'YA', 'YE/YI', 'YO/YU', 'Y', 'RA', 'RE/RI', 'RO/RU', 'R']
all_chars.reverse() #revese the all_chars array because it has been encoded here in reverse

# default route
@app.route('/')
def index():
	return "Index API"

	# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
	return """
	Wrong URL!
	<pre>{}</pre>""".format(e), 404

@app.errorhandler(500)
def server_error(e):
	return """
	An internal error occurred: <pre>{}</pre>
	See logs for full stacktrace.
	""".format(e), 500

# API route
@app.route('/classify-image', methods=['POST'])
def api():
	#get the input
#try:
	imgdata = base64.b64decode(request.form['imageData'])
	img = cv2.imdecode(np.fromstring(imgdata, np.uint8), cv2.IMREAD_GRAYSCALE)
	img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
	img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
	#res_float, res, all_res, res_char = classifier(img)
	res_float, res, all_res, res_char = model_api(img)
	#response = jsonify(output_data)
	data = {
		'status':1,
		'message':'Image classification complete.', 
		'result':res.tolist(), 
		'result_float':res_float.tolist(), 
		'char':res_char, 
		'all_chars':all_chars
	}
    js = json.dumps(data)
    resp = Response(js, status=200, mimetype='application/json')
    return resp
	#return jsonify({'status':1, 'message':'Image classification complete.', 'result':res.tolist(), 'result_float':res_float.tolist(), 'char':res_char, 'all_chars':all_chars})
#except:
#	return jsonify({'status': -1, 'message': 'Probably not an image!'})

@app.route('/test-upload')
def testUpload():
	return '<form action="/classify-image1" method="post" enctype="multipart/form-data"><input type="file" name="imageData" /><button type="submit">Upload</button></form>'

@app.route('/classify-image1', methods=['POST'])
def classifyImage1():
	img = cv2.imdecode(np.fromstring(request.files['imageData'].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
	img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
	img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
	#res_float, res, all_res, res_char = classifier(img)
	res_float, res, all_res, res_char = model_api(img)
	return jsonify({'status':1, 'message':'Image classification complete.', 'result':res.tolist(), 'result_float':res_float.tolist(), 'char':res_char, 'all_chars':all_chars})
	try:
		print("TEST")
	except:
		return jsonify({'status': -1, 'message': 'Probably not an image!'})

if __name__ == '__main__':
	app.run(debug=True, port=8080, host='0.0.0.0')