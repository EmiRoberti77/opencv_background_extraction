import cv2
import numpy as np

mask_output = "mask_output.mp4"
diff_output = "diff_output.mp4"
cap = cv2.VideoCapture("cars_highway.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# grab the first frame as reference
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

mask_out = cv2.VideoWriter(mask_output, fourcc, 24, (first_gray.shape[1], first_frame.shape[0]))
diff_out = cv2.VideoWriter(diff_output, fourcc, 24, (first_gray.shape[1], first_frame.shape[0]))

mog2 = cv2.createBackgroundSubtractorMOG2()

while True:
  ret, frame = cap.read()
  if not ret:
    break

  mask = mog2.apply(frame)
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  diff = cv2.absdiff(first_gray, gray)
  # display the frames
  cv2.imshow("Image",diff)
  cv2.imshow("Mask",mask)

  # save the frames to file and convert back to BGR as the video writer needs a 3 channel frame
  diff_out.write(cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR))
  mask_out.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))


  # store previous frame so its compared to new frame on next loop
  first_gray = gray

  key = cv2.waitKey(1)
  if key == ord('q'):
    break

diff_out.release()
mask_out.release()
cap.release()
cv2.destroyAllWindows()

