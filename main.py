import glob
import cv2

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
            k -= 0x100000
            if k == 27 or k == 113:
                break

def process_video():
    pipeline = LanePipeline()

    clip1 = VideoFileClip('project_video.mp4')
    video_clip = clip1.fl_image(pipeline.process)
    video_clip.write_videofile('output_video.mp4', audio=False)

if __name__ == "__main__":
    process_video()

