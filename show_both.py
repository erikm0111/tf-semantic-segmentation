import cv2

file_path = 'video/t7.mp4'
file_path2 = 'video/prediction.avi'

cap = cv2.VideoCapture(file_path)
cap2 = cv2.VideoCapture(file_path2)

while cap.isOpened():
	ret, real_frame = cap.read()
	ret2, mask_frame = cap2.read()

	alpha = 0.35
	blended = cv2.addWeighted(real_frame, alpha, mask_frame, (1 - alpha), 0.0)
	height, width = blended.shape[:2]
	res = cv2.resize(blended, (int(width/1.8), int(height/1.8)), interpolation = cv2.INTER_CUBIC)
	cv2.imshow('blended', res)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cap2.release()
cv2.destroyAllWindows()
