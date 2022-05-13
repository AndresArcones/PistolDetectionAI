from flask import Flask, request, jsonify, send_file
import imageio
import torch
import cv2
import boto3
import uuid


app = Flask(__name__)


def round_int(x):
    if x in [float("-inf"), float("inf")]:
        return float("nan")
    return int(round(x))


@app.route("/mostrar-detecciones", methods=['GET'])
def mostrarDetecciones():
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
            if frame_number % 30 == 0:
                result = model(im)
                if(result.pandas().xyxy[0].size > 0):
                    detections.append(
                        result.pandas().xyxy[0].iloc[0].values.tolist())

        resultados_deteccion = []
        for values in detections:
            resultados_deteccion.append(
                {"x1": values[0], "y1": values[1], "x2": values[2], "y2": values[3], "confianza": values[4], "detectado": values[6]})

    return jsonify("detecciones", resultados_deteccion)


@app.route("/detect", methods=['GET'])
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

        result_img = []
        for frame_number, im in enumerate(reader):
            # im is numpy array

            if frame_number % 30 == 0:
                print(frame_number)
                result = model(im)
                if(result.pandas().xyxy[0].size > 0):
                    if(result.pandas().xyxy[0].iloc[0].values.tolist()[4] > 0.7):
                        result_img = cv2.rectangle(im, (int(result.pandas().xyxy[0].iloc[0].values.tolist()[0]), int(result.pandas().xyxy[0].iloc[0].values.tolist()[1])), (int(result.pandas().xyxy[0].iloc[0].values.tolist()[2]), int(result.pandas().xyxy[0].iloc[0].values.tolist()[3])),
                                                   (255, 0, 0), 3)

        if(result_img != []):
            img_serial = cv2.imencode('.png', result_img)[1].tostring()
            s3 = boto3.resource('s3')
            key = str(uuid.uuid4()) + '.png'
            s3.Bucket("proyecto-fin-de-grado").put_object(Key=key,
                                                          Body=img_serial, ContentType='image/png', ACL='public-read')

            url = f'https://proyecto-fin-de-grado.s3.amazonaws.com/{key}'

    return url


# debe de ejecutarse al final
if __name__ == '__main__':
    app.run(debug=True, port=5000)
