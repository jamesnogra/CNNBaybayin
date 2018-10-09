from flask import Flask, request, jsonify
from flask_cors import CORS
from serve import get_model_api  # see part 1.

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default
model_api = get_model_api()

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
@app.route('/api', methods=['POST'])
def api():
	#get the input
	try:
		imgdata = base64.b64decode(request.form['imageData'])
		img = cv2.imdecode(np.fromstring(imgdata, np.uint8), cv2.IMREAD_GRAYSCALE)
		img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] # convert image to black and white pixels
		img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
		#res_float, res, all_res, res_char = classifier(img)
		res_float, res, all_res, res_char = model_api(img)
		#response = jsonify(output_data)
		return response
		return jsonify({'status':1, 'message':'Image classification complete.', 'result':res.tolist(), 'result_float':res_float.tolist(), 'char':res_char, 'all_chars':all_chars})
	except:
		return jsonify({'status': -1, 'message': 'Probably not an image!'})