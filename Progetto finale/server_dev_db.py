import os.path
import sys
from pathlib import Path

from flask import Flask, render_template, request
import inference
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Insert the path to this file
base_dir = Path('/home/michel/micheal_tirocinio/Progetto finale')
sys.path.append((str(base_dir)))

class ServerDevDb:
    def __init__(self):
        # Set directories and flask
        self.template_dir = base_dir / 'templates'
        self.static_folder = base_dir / 'static'
        self.app = Flask(__name__, template_folder=self.template_dir, static_folder=self.static_folder)

        # === MIGRATION CONFIGURATION ===
        CORS(self.app)
        self.UPLOAD_FOLDER = base_dir / 'static/uploads'
        self.app.config['UPLOAD_FOLDER'] = self.UPLOAD_FOLDER
        self.uploaded_file_path = None
        # container of list for image uploaded recently
        self.last_uploaded_file = []
        self.ALLOWED_EXTENTIONS = {'png', 'jpg', 'jpeg'}
        # Give the file of the classes of your model
        self.path_to_classes = base_dir / '5_foods_classes.txt'
        self.image_path = base_dir / 'Images'
        # Give the absolute path of your model weights
        self.path_to_weights = '/home/michel/models/effnetb2_5_foods_v1.pt'
        self.flask_path = base_dir / 'static/uploads'
        if not os.path.exists(self.UPLOAD_FOLDER):
            os.makedirs(self.UPLOAD_FOLDER, exist_ok=True)
        # main page HTML, deppl learning inference
        self.app.add_url_rule('/Progetto-finale/dl-image-inference', view_func=self.dl_image_inference, methods=['GET', 'POST'])

    # ALLOWED FILES - check extensions of the file to do inference
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENTIONS

    def dl_image_inference(self):
        # the method for make inference
        if request.method == 'POST':
            file = request.files['image_name']
            if file and self.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                # get file name
                self.last_uploaded_file.append(filename)
                if not os.path.exists(self.app.config['UPLOAD_FOLDER']):
                    os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
                file_path = os.path.join(self.app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                print(f'Uploaded File: {file}')
                # inference, pass the path to classes, the file to do inference and model's weights
                inference_effnet = inference.pred_image(self.path_to_classes, file_path, self.path_to_weights)
                # get back the image uploaded for inference and the string with the results
                return render_template('dl_image_inference.html', upload=True,
                                        image=filename, result=inference_effnet)
        # if there are errors in the upload do nothing
        return render_template('dl_image_inference.html', upload=False)


if __name__ == '__main__':
    # start flask server
    model_flask = ServerDevDb()
    model_flask.app.run(debug=True)
