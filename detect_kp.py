from audioop import avg
import PIL
import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import (
    draw_keypoints,
)  # Ezt kimásoltam a hivatalos pytorch oldalról, mert nem jött le a poetry-vel együtt ez a fgv, ki lett véve a 0.11.1-s verzióból
from torchvision.ops import box_area


import logging
import cv2
from enum import IntEnum
import time


class DetectKP:
    """
    DetectKP class returns a pretraind Resnet50 person keypointdetection pytorch model.

    Attributes
    ----------
    model: class 'torchvision.models.detection.keypoint_rcnn.KeypointRCNN'
        The pytorch Resnet50 keypoint rcnn model

    thresh: float
        A float int the [0;1] interval, the thershold of the donfidenco score in model prediction.

    img_tens: torch.Tensor
        The input image as a pytorch FloatTensor.

    out: list
        The output of the inference. It includes the keypoint positions,keypoint scores , the confidence scores of the persons, the bboxes,
        and labels (this is always 1, since the model can only recognize people).

    connect_skeleton: list containing (1,2) shaped tuples
        A list that indicates which two keypoints are connected. The order of the keypoints in the used mode is:
            coco_keypoints = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle",
            ]

    Methods
    -------
    load_data(source: str, path: str = "")
        Loads the data based on the input.

    evaluate()
        Makes an evaluation on the loaded image/images

    draw( save: bool = False):
        Draws the predicted keypoint to the image. Can also save it.
    """

    def __init__(self, thresh: float):
        """
        Loads the keypoint detection model, and sets up the threshhold for the confidence score.

        Parameters
        ----------
        thresh(float):
            The threshhold for the confidence score.
        """
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained=True
        )
        self.thresh = thresh
        self.model.cuda()
        self.model.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.out = {"all": None, "keypoints": None, "Scores": None, "bbox": None}
        self.connect_skeleton = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 4),
            (0, 5),
            (0, 6),
            (5, 7),
            (6, 8),
            (7, 9),
            (8, 10),
            (5, 11),
            (6, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
        ]
        self.coco_keypoints = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        self.bbox = None

    def load_img_data(self, path: str = ""):
        """
        Loads the image data based on its source.

        Parameters
        ----------
        path(str)
            The path of the image library.
        """

        img_int = Image.open(path)  # Beolvasás
        img = img_int.convert("RGB")  # Conversion to float
        transf = torchvision.transforms.Compose([T.ToTensor()])
        self.img_tens = transf(img)
        self.img_tens = self.img_tens.to(self.device)  # Adat GPU-ra

    def load_vid_data(self, frame: np.array):
        transf = torchvision.transforms.Compose([T.ToTensor()])
        self.img_tens = transf(frame)
        self.img_tens = self.img_tens.to(self.device)

    def evaluate(self):
        """
        Makes an evaluation, and creates self.outs, which contains the evaluation data.
        """
        self.out["all"] = self.model([self.img_tens])[0]
        self.out["keypoints"] = self.out["all"]["keypoints"]
        self.out["scores"] = self.out["all"]["scores"]
        self.out["bbox"] = self.out["all"]["boxes"]

    def draw(self, save: bool = False, path=""):
        """
        Draws the keypoints to the image.

        Parameters
        ----------
        save(bool):
            If true, is saves the image to the path given

        path(str):
            The path for saving the image.
        """

        ind = torch.where(self.out["scores"] > self.thresh)
        self.img_tens = self.img_tens.cpu()
        res = draw_keypoints(
            (self.img_tens * 255).type(torch.uint8),
            self.out["keypoints"][ind],
            colors="blue",
            connectivity=self.connect_skeleton,
            width=3,
        )

        img = res.permute(1, 2, 0).numpy()

        # if vid == False:
        #   cv2.imshow("img", res.permute(1, 2, 0).numpy())
        #   cv2.waitKey(0)
        #   if save == True:
        #       plt.savefig(path)
        # if vid == True:
        cv2.imshow("vid", img)


class Poses(IntEnum):

    """
    Pózokra enum
    """

    COMMAND_TOGGLE = 0
    X = 1
    Y = 2
    Z = 3
    

class Angles(IntEnum):

    """
    A keresett szögek
    """

    LEFT_ARMPIT = 0
    RIGHT_ARMPIT = 1
    LEFT_ELBOW = 2
    RIGHT_ELBOW = 3
    LEGS = 4


class Joints(IntEnum):

    """
    A keypointokra enum
    """

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class Control(DetectKP):

    """
    Subclass of DetectKP. It is used for recognizing the commands given, based on hand posture, and toggles if it receives commands, or
    not.

    Methods
    -------


    Parameters
    -----------
    _coco_keypoints: list
        Dont change this, it shows the order of the keypoints in the model outputs.

    current _pose: str
        Its value is the current handposition detected based on the kps.

    receive_commands: bool
        Whether the model listens to the commands detected.

    joint_positions: torch tensor


    torso_positions: dict
        Contains the name of the positions, and a number value assigned to it.

    """

    def __init__(self, thresh: float):
        super().__init__(thresh)

        self.current_pose = [False]
        for i in range(3):
            self.current_pose.append(0)
        self.receive_commands = False

        self.joint_positions = self.out["keypoints"]
        self.vectors = np.zeros((1, 5))  # The positions of arms, and the torso
        self.angles = np.zeros((1, 4))  # Angles

        self.t = []  # Saves time values

        self.prev_bbox_size = None

        self.velocity = None
        self.bboxes = self.out["bbox"]

    def _calculate_angles(self, vecs: np.array):
        """
        A function that calculates the angles that are used for classyifing a pose. The input is a numpy array,
        which contains the following joints, and must be int this order: Torso,left shouldes, right shoulder, left elbow, right elbow

        Returns
        --------
        self.angles: numpy array


        """

        tangents = np.array([np.arctan2(vecs[:, 0], vecs[:, 1])])[0]

        self.angles = np.abs(
            np.array(
                [
                    tangents[0] - tangents[2],  # Bal váll
                    tangents[1] - tangents[3],  # Jobb váll
                    tangents[2] - tangents[4],  # Bal könyök
                    tangents[3] - tangents[5],  # Jobb könyök
                    tangents[6] - tangents[7],  # Terpesz szöge
                ]
            )
        )

    def _tensor_to_array(self, tensor):
        """
        Transforms torch.tensor which is grad enabled, and on GPU, to a numpy array.
        """
        arr = tensor.cpu().detach().numpy()
        return arr

    def detect_pose(self, frame):
        """
        Calculate the angle between body and upper arm, and upper arm and lower arm.

        Parameters
        -----------

        frame: any type that contains image data
            The image to be processed
        """

        self.load_vid_data(frame)
        self.evaluate()
        try:  # Ha nem ismert fel semmit, akkor indexerrort dobott, ezt küszöböltem ki.
            self.joint_positions = self.out["keypoints"][0]
            self.bboxes = self.out["bbox"][0]

            # Megfelelő vektorok origóba eltolása. Az elnevezések azt mondják meg,
            # hogy hova mutat.

            left_hip = (
                self.joint_positions[Joints.LEFT_HIP][0:2]
                - self.joint_positions[Joints.LEFT_SHOULDER][0:2]
            )

            right_hip = (
                self.joint_positions[Joints.RIGHT_HIP][0:2]
                - self.joint_positions[Joints.RIGHT_SHOULDER][0:2]
            )

            right_elbow = (
                self.joint_positions[Joints.RIGHT_ELBOW][0:2]
                - self.joint_positions[Joints.RIGHT_SHOULDER][0:2]
            )

            left_elbow = (
                self.joint_positions[Joints.LEFT_ELBOW][0:2]
                - self.joint_positions[Joints.LEFT_SHOULDER][0:2]
            )
            left_wrist = (
                -self.joint_positions[Joints.LEFT_WRIST][0:2]
                + self.joint_positions[Joints.LEFT_ELBOW][0:2]
            )
            right_wrist = (
                -self.joint_positions[Joints.RIGHT_WRIST][0:2]
                + self.joint_positions[Joints.RIGHT_ELBOW][0:2]
            )

            left_leg = (
                self.joint_positions[Joints.LEFT_ANKLE][0:2]
                - self.joint_positions[Joints.LEFT_HIP][0:2]
            )
            right_leg = (
                self.joint_positions[Joints.RIGHT_ANKLE][0:2]
                - self.joint_positions[Joints.RIGHT_HIP][0:2]
            )

            # Tenzor->np.array

            self.vectors = np.array(
                [
                    self._tensor_to_array(left_hip),
                    self._tensor_to_array(right_hip),
                    self._tensor_to_array(left_elbow),
                    self._tensor_to_array(right_elbow),
                    self._tensor_to_array(left_wrist),
                    self._tensor_to_array(right_wrist),
                    self._tensor_to_array(left_leg),
                    self._tensor_to_array(right_leg),
                ]
            )

            # Szögek

            self._calculate_angles(self.vectors)
            self.angles = np.rad2deg(self.angles)

            # Command toggle

            if 20 < self.angles[Angles.LEGS]:

                self.current_pose[Poses.COMMAND_TOGGLE]=True
                self.t.append(time.time())

                if self.receive_commands == False:
                    if self.t[-1] - self.t[0] >= 3:
                        self.receive_commands = True

            else:
                self.receive_commands = False
                

            if self.angles[Angles.LEGS] < 20:
                self.t = []
                self.current_pose[Poses.COMMAND_TOGGLE]=False
                self.current_pose[Poses.X]=0
                self.current_pose[Poses.Y]=0
                self.current_pose[Poses.Z]=0

            # Pózok

            if self.receive_commands == True:

                for i in range(3):
                    self.current_pose[i+1]=0

                if 100 < self.angles[Angles.LEFT_ARMPIT] <= 160:
                    self.current_pose[Poses.X]=1

                if 20 < self.angles[Angles.LEFT_ARMPIT] <= 80:
                    self.current_pose[Poses.X]=-1

                if 100 < self.angles[Angles.RIGHT_ARMPIT] <= 160:             
                    self.current_pose[Poses.Y]=1

                if  20 < self.angles[Angles.RIGHT_ARMPIT] <= 80:
                    self.current_pose[Poses.Y]=-1

                if 100>self.angles[Angles.LEFT_ELBOW]>0:
                    self.current_pose[Poses.Z]=-1

                if 100>self.angles[Angles.RIGHT_ELBOW]>0:
                    self.current_pose[Poses.Z]=1



        except IndexError:
            self.joint_positions = None


    def draw(self, bbox: bool = True):
        """
        Draws the keypoints to the image.

        Parameters
        ----------
        save(bool):
            If true, is saves the image to the path given

        path(str):
            The path for saving the image.
        """

        ind = torch.where(self.out["scores"] > self.thresh)
        self.img_tens = self.img_tens.cpu()
        res = draw_keypoints(
            (self.img_tens * 255).type(torch.uint8),
            self.out["keypoints"][ind],
            colors="blue",
            connectivity=self.connect_skeleton,
            width=3,
        )

        

        self.img = res.permute(1, 2, 0).numpy()

        try:  # A szögek, és a jelenlegi utasítás kiírása, ebből majd lehet 1-2 dolog ki lesz szedve vagy
            #  kiszépítem kicsit.
            cv2.putText(
                self.img,
                "Current pose: " +str(self.current_pose[Poses.COMMAND_TOGGLE])+','+ str(self.current_pose[Poses.X])+','+str(self.current_pose[Poses.Y])+','+str(self.current_pose[Poses.Z]),
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 0, 0),
                thickness=2,
            )

            cv2.putText(
                self.img,
                "Receive_commands: " + str(self.receive_commands),
                (350, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 0, 0),
                thickness=2,
            )
            '''
            for angs in Angles:
                cv2.putText(
                    self.img,
                    f"{angs.name}: {self.angles[angs.value]:.1f}",
                    (25, 600 + angs.value * 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    thickness=2,
                )
            '''
            text_positions=[self.joint_positions[Joints.LEFT_SHOULDER],
                            self.joint_positions[Joints.RIGHT_SHOULDER],
                            self.joint_positions[Joints.LEFT_ELBOW],
                            self.joint_positions[Joints.RIGHT_ELBOW],
                            self.joint_positions[Joints.RIGHT_HIP]
            ]

            for angs in Angles:
                cv2.putText(
                    self.img,
                    str(round(self.angles[angs.value],1)),
                    (int(text_positions[angs.value][0]), int(text_positions[angs.value][1])+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=(0, 0, 255),
                    thickness=2,
                )
            # if bbox == True:
            #   torchvision.utils.draw_bounding_boxes(
            #       self.img_tens,
            #       self.bboxes,
            #   )

        except TypeError as msg:
            print(msg)
            pass

        cv2.imshow("vid", self.img)