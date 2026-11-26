import cv2
import imutils
from imutils.contours import sort_contours
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

st.title("Handwriting Recognition (OCR)")
st.write("Handwritten letters aur numbers (A-Z, 0-9) wali image upload karein.")


@st.cache_resource
def load_ocr_model():
  return load_model("handwriting.model")


model = load_ocr_model()
labelNames = [l for l in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

uploaded_file = st.file_uploader(
    "Image choose karein...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)

  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  edged = cv2.Canny(blurred, 30, 150)

  cnts = cv2.findContours(
      edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  cnts = imutils.grab_contours(cnts)
  cnts = sort_contours(cnts, method="left-to-right")

  chars = []
  for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    if (5 <= w <= 150) and (15 <= h <= 120):
      roi = gray[y : y + h, x : x + w]
      thresh = cv2.threshold(
          roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
      )
      (tH, tW) = thresh.shape
      thresh = (
          imutils.resize(thresh, width=32)
          if tW > tH
          else imutils.resize(thresh, height=32)
      )
      (tH, tW) = thresh.shape
      dX, dY = int(max(0, 32 - tW) / 2.0), int(max(0, 32 - tH) / 2.0)
      padded = cv2.copyMakeBorder(
          thresh, dY, dY, dX, dX, cv2.BORDER_CONSTANT, value=(0, 0, 0)
      )
      padded = (
          np.expand_dims(cv2.resize(padded, (32, 32)), axis=-1).astype(
              "float32"
          )
          / 255.0
      )
      chars.append((padded, (x, y, w, h)))

  if len(chars) > 0:
    boxes = [b for b in chars]
    chars = np.array([c for c in chars], dtype="float32")
    preds = model.predict(chars)

    detected_text = ""
    for pred, (x, y, w, h) in zip(preds, boxes):
      label = labelNames[np.argmax(pred)]
      detected_text += label
      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.putText(
          image,
          label,
          (x - 10, y - 10),
          cv2.FONT_HERSHEY_SIMPLEX,
          1.2,
          (0, 255, 0),
          2,
      )

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Detected Text",
        use_container_width=True,
    )
    st.success(f"Detected Text: **{detected_text}**")
  else:
    st.warning("Image mein koi clear character nahi mila.")
