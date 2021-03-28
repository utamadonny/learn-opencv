import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

cascPath = "Face Detection/haarcascade_frontalface_alt.xml"
cascPath2 = "Face Detection/haarcascade_eye.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(cascPath2)
log.basicConfig(filename='webcam.log',level=log.INFO)

{
  // "python.pythonPath": "C:\\Users\\utama\\AppData\\Local\\Programs\\Python\\Python39\\python.exe",
  "editor.tokenColorCustomizations": {
    "textMateRules": [{
        "scope": "meta.function-call.generic.python",
        "settings": {
          "foreground": "#FF0000"
        }
      },
      {
        "scope": "source.python",
        "settings": {
          "foreground": "#000000"
        }
      },
      {
        "scope": "punctuation.definition.string.begin",
        "settings": {
          "foreground": "#3dc15c"
        }
      },
      {
        "scope": "punctuation.definition.string.end",
        "settings": {
          "foreground": "#3dc15c"
        }
      },
      // {
      //   "scope": "punctuation.definition.comment.python",
      //   "settings": {
      //     "foreground": "#008000"
      //   }
      // },
      // {
      //   "scope": "punctuation",
      //   "settings": {
      //     "foreground": "#0202ff"
      //   }
      // },
      {
        "scope": "support.function.builtin.python",
        "settings": {
          "foreground": "#0000FF"
        }
      },
      {
        "scope": "constant.numeric",
        "settings": {
          "foreground": "#99048d"
        }
      },
      // {
      //   "scope": "meta.function-call.python",
      //   "settings": {
      //     "foreground": "#ff0080"
      //   }
      // },
      {
        "scope": "keyword.operator",
        "settings": {
          "foreground": "#05979c"
        }
      },
      {
        "scope": "meta.attribute.python",
        "settings": {
          "foreground": "#ff0000"
        }
      }
    ],



    //     "[GitHub Light]": {
    //     "textMateRules": [
    //       {
    //         "scope": "meta.function-call.generic.python",
    //         "settings": {
    //           "foreground": "#FF0000"
    //         }
    //       },
    //       {
    //         "scope": "source.python",
    //         "settings": {
    //           "foreground": "#000000"
    //         }
    //       },
    //       {
    //         "scope": "punctuation.definition.string.begin",
    //         "settings": {
    //           "foreground": "#3dc15c"
    //         }
    //       }, {
    //         "scope": "punctuation.definition.comment.python",
    //         "settings": {
    //           "foreground": "#3dc15c"
    //         }
    //       },
    //       {
    //         "scope": "punctuation.definition.string.end",
    //         "settings": {
    //           "foreground": "#3dc15c"
    //         }
    //       },
    //       {
    //         "scope": "punctuation",
    //         "settings": {
    //           "foreground": "#e5ff73"
    //         }
    //     }
    //     ]
    //   }
    //  }
  }
video_capture = cv2.VideoCapture(0)
anterior = 0

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    eyes  = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" eyes: "+str(len(eyes))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

# Reference : 
#https://github.com/shantnu/Webcam-Face-Detect/blob/master/webcam.py