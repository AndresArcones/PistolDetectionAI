from flask import Flask, request, jsonify
import imageio
import torch


app = Flask(__name__)


def round_int(x):
    if x in [float("-inf"), float("inf")]:
        return float("nan")
    return int(round(x))


@app.route("/mostrar-decciones", methods=['GET'])
def detect():
    if not request.method == "GET":
        return "error"

    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='.\Pesos_IA\YOLOv5_416x416_No_augmentation_100_epochs.pt', force_reload=True)
    videoURL = request.args.get('videoURL')

    if videoURL != '':
        print(videoURL)
        reader = imageio.get_reader(
            videoURL, 'ffmpeg')
        totalFrames = reader._meta['nframes']
        print(len(reader))

        detections = []
        for frame_number, im in enumerate(reader):
            # im is numpy array
            if frame_number % 32 == 0:  # round(totalFrames*0.01)
                result = model(im)
                if(result.pandas().xyxy[0].size > 0):
                    detections.append(
                        result.pandas().xyxy[0].iloc[0].values.tolist())

        resultados_deteccion = []
        for values in detections:
            resultados_deteccion.append(
                {"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3], "confianza": values[4], "detectado": values[6]})

    return jsonify("detecciones", resultados_deteccion)


# debe de ejecutarse al final
if __name__ == '__main__':
    app.run(debug=True, port=5000)
