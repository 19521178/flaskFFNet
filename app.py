import json
from flask import Flask, session, request, render_template
from flask_restful import reqparse, abort, Api, Resource
from flask_cors import CORS

import os
import base64
from io import BytesIO
from PIL import Image

# from model.layers import create_FC_layers
# import numpy as np
# from model.neural_network import FCNet, GRUNet
# from model.post_processor import ExtendBoth
# from model.utils.DictNamespace import DictNamespace as dict
# from model.extract_feature_model import AlexNet, MobileNetV3Large
# from model.static_var import *
# from model.post_processor import ExtendAll, NoneExtend

# opt_params = dict(
#     optimizer = TypesOptimizer.RMSPROP,
#     lr = 1e-4,
#     loss = TypesLoss.MEAN_SQUARE
# )
# model_params = dict(
#     dim_input = 4096,
#     dim_hidden = 256,
#     dim_output = 25
# )
# regular_params = None
# type_model = TypesModel.FC
# dim_input = 4096
# dims_FC = [400, 200, 100, 25]
# FC_layers = create_FC_layers([dim_input] + dims_FC, is_regular=False, regular_params = regular_params)
# model = FCNet(opt_params, model_params, FC_layers)
# model.load_model('./model/', 'epoch_80')

# post_processor = ExtendBoth(
#     window_size = 0,
#     step_select = 1,
#     window_size_skip = 4,
#     step_select_skip = 3,
#     threshold = 15
# )

# extractor = MobileNetV3Large()

current_frame = None
# previous_info = model.init_info()

# app = Flask(__name__, template_folder='client/', static_folder='client/')
app = Flask(__name__, template_folder='client', static_folder='client')
app.config['UPLOAD_FOLDER'] = '/uploads'
app.secret_key = '@nChlnh'
api = Api(app)
CORS(app)


def abort_if_frame_doesnt_exist():
    if current_frame is None:
        abort(404, message="There's not any frame has uploaded")

# parser = reqparse.RequestParser()
# parser.add_argument('image')

class CurrentFrame(Resource):
    def get(self):
        abort_if_frame_doesnt_exist()
        return current_frame
    
    def post(self):
        # print("form: ", (request.form))
        # print("data: ", (request.data[22:30]), request.data[-10:])
        # print("values: ", (request.values))
        # print("args: ", (request.args))
        # print("files: ", (request.files))
        # print('method: ', request.method)
        # print('header: ', request.headers)
        # print('get_data: ', request.get_data())
        # print('cookies: ', request.cookies)
        # print('content_encode: ', request.content_encoding)
        
        content = request.data[22:]
        #  + b'=' * (-len(content) % 4)
        im_bytes = base64.b64decode(content)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        img = Image.open(im_file)
        current_frame = img
        print(img)
        
        # curr_feat = extractor.forward(current_frame)
        # curr_info, a_value = model.forward(previous_info, curr_feat)
        # previous_info = curr_info
        # action = np.argmax(a_value) + 1
        # indices_neighbor = post_processor.formula(action)
        # return {'action': action, 'indices neighbor': indices_neighbor}, 201
        return {'action': 1, 'indices neighbor': [0]}, 201
#
# class TodoList(Resource):
#     def get(self):
#         return TODOS

#     def post(self):
#         args = parser.parse_args()
#         todo_id = int(max(TODOS.keys()).lstrip('todo')) + 1
#         todo_id = 'todo%i' % todo_id
#         TODOS[todo_id] = {'task': args['task']}
#         return TODOS[todo_id], 201

##
## Actually setup the Api resource routing here
##
api.add_resource(CurrentFrame, '/frame')
# api.add_resource(TodoList, '/todos')

@app.route('/')
def login():
    # GHI DANH AUTHORIZE
    # lưu session, user_id;
    # khởi tạo biến current_frame và previous_info cho user;
    # trả về token
    print(os.listdir('./client'))
    print(app.template_folder)
    # return render_template('client/index.html')
    return render_template(template_name_or_list='index.html')

if __name__ == '__main__':  
    app.run(debug=True)