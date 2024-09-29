import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

class CaptureFrames:
    def __init__(self, model, show_mask=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.show_mask = show_mask

    def __call__(self, image):
        return self.process_image(image)

    def process_image(self, image):
        img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if isinstance(image, np.ndarray):
            orig = image
        else:
            raise TypeError("Expected input to be a NumPy array")

        shape = orig.shape[0:2]
        frame = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256), cv2.INTER_LINEAR)

        a = img_transform(Image.fromarray(frame))
        a = a.unsqueeze(0)
        imgs = Variable(a.to(dtype=torch.float, device=self.device))
        self.model.eval()
        pred = self.model(imgs)

        pred = torch.nn.functional.interpolate(pred, size=[shape[0], shape[1]])
        mask = pred.data.cpu().numpy().squeeze()
        mask = mask > 0.8

        # Create blurred image
        blurred_image = cv2.GaussianBlur(orig, (15, 15), 0)

        # Find contours and create a mask for the contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        face_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(face_mask, contours, -1, 255, thickness=cv2.FILLED)

        # Apply mask to the original image
        masked_face = cv2.bitwise_and(orig, orig, mask=face_mask)

        # Create a mask for blurring
        mask_inv = np.logical_not(face_mask).astype(np.uint8)
        blurred_background = cv2.bitwise_and(blurred_image, blurred_image, mask=mask_inv)

        # Final output with blurred background
        final_output = cv2.add(masked_face, blurred_background)

        # Draw contours on the final output for visualization
        final_output_with_contours = final_output.copy()

        return final_output_with_contours, face_mask
