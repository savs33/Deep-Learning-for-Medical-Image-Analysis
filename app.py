#python3.5

# from gevent import monkey; monkey.patch_all()

import os,sys,json
from bottle import route, run, static_file, template, view, Response
from bottle import get, post, request
import process_xray
from cam import Visualizer
from unet import Segmenter
import cv2



@route('/css/<filename>')
def img_static(filename):
	return static_file(filename, root='./static/css')

@route('/img/<filename>')
def img_static(filename):
	return static_file(filename, root='./static/img')

@route('/js/<filename>')
def js_static(filename):
	return static_file(filename, root='./static/js')

@route('/fonts/<filename>')
def js_static(filename):
	return static_file(filename, root='./static/fonts')

@route('/tmp_imgs/<filename>')
def js_static(filename):
	return static_file(filename, root='./static/tmp_imgs')

@route("/")
@view("upload")
def hello(content=''):
	if content == '':
		return dict(disp_msg='Please upload an image and click on the operation to be performed')
	else:
		return dict(disp_msg=content)

@route('/',method="POST")
def upload_process():
	if request.method == "POST":

		if request.forms.get('tb_submit_button'):
			data = request.files.xray_img
			if data and data.file:
				return display_tb_result(data)
			else:
				return hello(content='Image not specified. Upload an image before clicking the button')

		elif request.forms.get('seg_submit_button'):
			data = request.files.xray_img
			if data and data.file:
				return display_seg_result(data)
			else:
				return hello(content='Image not specified. Upload an image before clicking the button')

		else:
			return hello()

@route("/detection_tb/")
@view('detection_tb')
def display_tb_result(data):
	raw_img = data.file.read() 
	filename = data.filename

	img = process_xray.convt_img(raw_img)
	img = cv2.resize(img,(512,512))
	img_path = 'static/tmp_imgs/1.png'
	status = cv2.imwrite(img_path,img)
	print(img_path,status)

	vis = Visualizer(model_path='models/TB_detection.h5')
	tb_percent = vis.save_tb_overlay(img_path,out_path='static/tmp_imgs/2.jpg')
	print(tb_percent)

	return dict(tb_percent=tb_percent)


@route("/seg_lungs/")
@view('detection_unet')
def display_seg_result(data):
	raw_img = data.file.read() 
	filename = data.filename

	img = process_xray.convt_img(raw_img)
	img = cv2.resize(img,(512,512))
	img_path = 'static/tmp_imgs/3.png'
	status = cv2.imwrite(img_path,img)
	print(img_path,status)

	seg = Segmenter(model_path='models/Unet_5.h5')
	out_path = 'static/tmp_imgs/4.jpg'
	seg.save_seg_mask(img_path,out_path=out_path)
	return dict(msg='Hover to see segmentation mask')



# -------------------MAIN START----------------------------
if __name__ == "__main__":
	try:
		port = int(sys.argv[1])
	except:
		port = 5000
		
	port = int(os.environ.get("PORT", port))
	run(
	host='localhost',
	port=port,
	debug=True,
	reloader = True
	)
# -------------------MAIN END----------------------------
