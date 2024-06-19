import cv2
import numpy as np
import torch


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(
    vid_list, fnum=8, target_size=(224, 224), device=torch.device("cuda")
):
    assert len(vid_list) >= fnum
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def frame_preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255
    return frame


def resize_frames(tensor, num_frames=8):
    """
    Resize the number of frames in a tensor to the specified number by sampling at regular intervals.

    Parameters:
    tensor (torch.Tensor): Input tensor of shape [batch, channel, num_frame, width, height]
    num_frames (int): Number of frames to resize to (default: 32)

    Returns:
    torch.Tensor: Tensor with the number of frames resized to num_frames by sampling
    """
    # Get the current shape of the tensor
    batch, channel, current_frames, width, height = tensor.shape

    if current_frames == num_frames:
        return tensor
    elif current_frames < num_frames:
        # Pad with zeros if the number of frames is less than the desired number
        pad_size = num_frames - current_frames
        padding = torch.zeros(
            (batch, channel, pad_size, width, height),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        resized_tensor = torch.cat((tensor, padding), dim=2)
    else:
        # Sample frames at regular intervals if the number of frames is greater than the desired number
        indices = torch.linspace(0, current_frames - 1, num_frames).long()
        resized_tensor = tensor[:, :, indices, :, :]

    return resized_tensor
