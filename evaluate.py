import numpy as np
import argparse
import yaml
import cv2
from ultralytics import YOLO
from videoDrawer import VideoDrawer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

NEW_WIDTH = 640  # Set your desired width
NEW_HEIGHT = 360  # Set your desired height

def main(groundTruth, resultFilePath):
    model = YOLO('Data/yolov8l-seg.pt')
    objsId = [configs["human_id"]] + list(configs["interactable_objs"].keys())

    groundTruthMasks = []
    videoDrawer = VideoDrawer(groundTruth, configs["output_folder_dir_path"])
    segClip = videoDrawer.CreateNewClip("segmentationOriginal", "", NEW_WIDTH, NEW_HEIGHT)

    while True:
        hasFrames, vidFrameData = videoDrawer.video.read() # gives in BGR format
        if not hasFrames:
            break

        resized_frame = cv2.resize(vidFrameData, (NEW_WIDTH, NEW_HEIGHT))

        results = model.predict(
            resized_frame, 
            conf = configs["yolo_params"]["confidence_score"],
            iou = configs["yolo_params"]["intersection_over_union"],
            classes = objsId
            )
        
        # Iterate each object contour 
        combined_mask = None
        for r in results:
            for c in r:
                # Create contour mask 
                contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                b_mask = np.zeros(resized_frame.shape[:2], np.uint8)
                if contour.any():
                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Combine contour mask with existing mask
                if combined_mask is None:
                    combined_mask = b_mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, b_mask)

        
        _, combined_mask_bw = cv2.threshold(combined_mask, 1, 255, cv2.THRESH_BINARY)
        combined_mask_bw = np.array(combined_mask_bw)
        

        #resized_mask = cv2.resize(combined_mask_bw, (videoDrawer.videoWidth, videoDrawer.videoHeight))
        mask_image = cv2.cvtColor(combined_mask_bw, cv2.COLOR_GRAY2BGR)
        groundTruthMasks.append(mask_image)
        segClip.write(mask_image)

    segClip.release()
    videoDrawer.StopVideo()

    resultsSegClip = videoDrawer.CreateNewClip("resultsSegmentation", "", NEW_WIDTH, NEW_HEIGHT)
    resultsMasks = []
    resultVid = cv2.VideoCapture(resultFilePath)
    MASK_HEIGHT, MASK_WIDTH = groundTruthMasks[0].shape[:2]
    while True:
        hasFrames, vidFrameData = resultVid.read() # gives in BGR format
        if not hasFrames:
            break

        resized_frame = cv2.resize(vidFrameData, (NEW_WIDTH, NEW_HEIGHT))

        # Convert frame to grayscale
        gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to separate objects from background
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

        # Invert the thresholded image
        inverted_thresh = cv2.bitwise_not(thresh)

        resized_mask = cv2.resize(inverted_thresh, (MASK_WIDTH, MASK_HEIGHT))
        mask_image = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        resultsMasks.append(np.array(mask_image))

        resultsSegClip.write(mask_image)
        
    resultsSegClip.release()
    resultVid.release()

    mask1_pixels = []
    mask2_pixels = []
    for index in range(0, len(resultsMasks)):
        frame1 = groundTruthMasks[index]
        frame2 = resultsMasks[index]

        # Flatten frames into 1D arrays
        mask1_pixels.append(frame1.flatten().astype(np.uint8))
        mask2_pixels.append(frame2.flatten().astype(np.uint8))

    # Convert lists to NumPy arrays
    mask1_pixels = np.array(mask1_pixels)
    mask2_pixels = np.array(mask2_pixels)
    print(len(mask1_pixels))
    print(len(mask2_pixels))
    print(len(mask1_pixels[0]))
    print(len(mask2_pixels[0]))

    # Compute confusion matrix
    conf_matrix = confusion_matrix(mask1_pixels.flatten(), mask2_pixels.flatten())
    
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]

    accuracy = (TN + TP) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = (2 * TP) / (2 * TP + FP + FN)

    print("Confusion matrix:")
    print(conf_matrix)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your application's description")
    parser.add_argument("--config", default='Configs/config.yaml', type=str, help="File path for config file")


    arguments = parser.parse_args()
    with open(arguments.config) as f:
        configs = yaml.safe_load(f)

    groundTruth = "Input/DropBallWalkForward.mp4"
    results = "Output/DropBallWalkForward/render.mp4"

    main(groundTruth, results)

    