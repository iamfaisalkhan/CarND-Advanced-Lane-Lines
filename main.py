import glob
import cv2
import sys

from pipeline import LanePipeline
from moviepy.editor import VideoFileClip

def process_images():
    images = glob.glob("test_images/test*.jpg")

    pipeline = LanePipeline()

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        result = pipeline.process(img)

        while (1):
            cv2.imshow('img', result)

            k = cv2.waitKey(100)
            
            if k == 27:
                break
            if k == 113:
                sys.exit(0)

def process_video():
    pipeline = LanePipeline()

    # cap = cv2.VideoCapture('project_video.mp4')
    # cap = cv2.VideoCapture('challenge_video.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('video.mp4', fourcc, 25, (1280, 720))

    while (cap.isOpened()):
        ret, frame = cap.read()

        result= pipeline.process(frame)
        result = cv2.resize(result, (1280, 720))
        cv2.imshow('img', result)

        # out.write(frame)

        while True:
            k = cv2.waitKey(25)
            if k == 27:
                sys.exit(0)
            elif k == ord('n'):
                break

def write_out_video(input, output):
    pipeline = LanePipeline()

    clip1 = VideoFileClip(input)
    video_clip = clip1.fl_image(pipeline.process)
    video_clip.write_videofile(output, audio=False)

if __name__ == "__main__":
    write_out_video(sys.argv[1], sys.argv[2])
