from glob import glob
import cv2
import skimage
import os
import numpy as np


class OdometryClass:
    def __init__(self, frame_path):

        self.frame_path = frame_path
        self.frames = sorted(
            glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))

        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]

        # set feature detector to be FastFeatureDetector
        self.detector = cv2.FastFeatureDetector_create()

        # initialize the first pose to set value of R and t
        initialize = np.array(self.pose[0]).reshape(3, 4).astype('float64')
        self.R = initialize[:3, :3]
        self.t = initialize[:, 3].reshape(3, 1)

    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors

        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)

    def detect(self, img):
        # get features with fast feature detector
        p0 = self.detector.detect(img)

        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def get_mono_coordinates(self):
        # finds monocoords so we can estimate position
        adj_coord = np.hstack([self.R, self.t]).reshape(-1, 1)

        return adj_coord

    def find_matrix(self, q1, q2, scale):
        # finds essential matrix
        E, mask = cv2.findEssentialMat(
            q2, q1, focal=self.focal_length, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(
            E, q2, q1, focal=self.focal_length, pp=self.pp)

        if (scale > 0.1):
            self.t += (scale * self.R.dot(t)).reshape(3, 1)
            self.R = R.dot(self.R)

    def optical_flow(self, previous_image, current_image):
        # calculates optical flow between previous and current image

        params = dict(winSize=(7, 7),
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        p1 = self.detect(previous_image)
        p2, st, _ = cv2.calcOpticalFlowPyrLK(
            previous_image, current_image, p1, None, **params)
        p1 = p1[st == 1]
        p2 = p2[st == 1]
        Good_points = np.hstack([p1, p2])

        q1 = Good_points[:, :2]
        q2 = Good_points[:, 2:]

        return q1, q2

    def run(self):
        num_images = len(self.pose)
        preds = np.zeros((num_images, 3))
        current_pose = self.pose[0]
        preds[0] = np.vstack(
            [np.array([float(current_pose[3]), float(current_pose[7]), float(current_pose[11])])])
        for i in range(num_images):

            if i != 0:
                old_frame = self.imread(self.frames[i-1])
                current_frame = self.imread(self.frames[i])
                scale = self.get_scale(i)
                q1, q2 = self.optical_flow(old_frame, current_frame)
                self.find_matrix(q1, q2, scale)
                pred = self.get_mono_coordinates()
                preds[i] = np.hstack([pred[3], pred[7], pred[11]])
                print(preds[i])

        return preds


if __name__ == "__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path, path.shape)
