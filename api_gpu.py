from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import FileResponse, JSONResponse
from uuid import uuid4
import os
from typing import Optional,Annotated
import subprocess
import uvicorn
import os
import shutil
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
import cv2
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from sam2.build_sam import build_sam2_video_predictor
import torch
sam2_checkpoint = "sam2_hiera_tiny.pt"
model_cfg = "sam2_hiera_t.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Define the directory where the video files will be saved
VIDEO_DIR = "videos"

# Create the video directory if it doesn't exist
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)


torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


BATCH_SIZE=3
@app.post("/upload_video")
async def upload_video(file: UploadFile  = File(...),batch_size:str=Form()):
    """
    Upload a video file, save it to a local directory using a unique ID as the directory name,
    and extract JPEG images from the video file using FFmpeg.

    Args:
        file (UploadFile): The video file to be uploaded.

    Returns:
        JSONResponse: A JSON response with the unique ID of the directory where the video file is saved.
    """
    print(batch_size)
    # Generate a unique ID for the directory
    unique_id = str(uuid4())+"_"+str(batch_size)
    # Create the directory for the video file
    video_dir = os.path.join(VIDEO_DIR, unique_id)
    os.makedirs(video_dir, exist_ok=True)

    # Save the video file to the directory
    file_path = os.path.join(video_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    # Extract JPEG images from the video file using FFmpeg
    image_dir = os.path.join(video_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    ffmpeg_command = f"ffmpeg -i {file_path} -q:v 1 -start_number 0  {image_dir}/%03d.jpg"
    subprocess.run(ffmpeg_command, shell=True)
    print(batch_size)
    move_img_to_batch_wise_folders(image_dir, os.path.join(video_dir,"img_batch"),batch_size=int(batch_size))
    # Return the unique ID of the directory
    return JSONResponse(content={"unique_id": unique_id}, status_code=200)


def move_img_to_batch_wise_folders(src_folder,dst_folder,batch_size):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Get a list of all image files in the source folder
    image_files = [f for f in os.listdir(src_folder) if f.endswith('.jpg')]

    # Sort the image files
    image_files.sort(key=lambda x: int(x.split('.')[0]))

    # Split the image files into batches of 200

    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    for i, batch in enumerate(batches):
        batch_folder = os.path.join(dst_folder, f'batch_{i}')
        os.makedirs(batch_folder, exist_ok=True)
        for image in batch:
            shutil.move(os.path.join(src_folder, image), batch_folder)

def find_batch(n, batch_size):
    print(n,batch_size)
    # Check if n is a positive integer
    if n <= 0:
        # raise ValueError("The number n must be a positive integer.")
        return 1,1
    
    # Calculate the batch number
    batch_number = (n - 1) // batch_size + 1
    index_within_batch = (n - 1) % batch_size + 1
    return batch_number,index_within_batch

def find_frame(batch_number, index_within_batch, batch_size):
    # Calculate the value of n
    n = (batch_number - 1) * batch_size + index_within_batch
    return n

@app.post("/process_frame")
async def process(data: dict):
    torch.cuda.empty_cache()
    batches={}
    ann_obj_id=1 # for now
    video_id=data['video_id']
    batch_size=int(video_id.split("_")[-1])
    frame_data=data["frame_data"]
    for i in frame_data:
        batch_number ,index_number= find_batch(int(i["frame"]), batch_size)
        i["index"] = index_number-1
        print(index_number )
        if not batches.get(batch_number):
            batches[batch_number] =[i]
        else:
            batches[batch_number].append(i)
    
    video_dir = os.path.join(VIDEO_DIR, video_id,"img_batch")
    if not os.path.exists(VIDEO_DIR):return JSONResponse(content={"status":"failed","status_msg":"id doesnot exist"})
    out_logits={}
    for  batch_no,batch in batches.items():
        ann_frame_idx=int(batch_no)
        video_batch_dir = f"{video_dir}/batch_{ann_frame_idx-1}/"
        print(video_batch_dir)
        inference_state = predictor.init_state(video_path=video_batch_dir)
        frame_data={ }
        for i in batch:
            if i["frame"] in frame_data:
                frame_data[i["frame"]] .append(i)
            else:
                frame_data[i["frame"]] = [i]
        for _,i in frame_data.items():
            cordinate=[[j["x"],j["y"]] for j in i]
            labels = np.array([1]*len(i), np.int32)
            points = np.array(cordinate ,dtype=np.float32)            
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=i[0]["index"],
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
                )
            out_logits[i[0]["frame"]]=out_mask_logits[0].cpu().numpy().tolist()
        # for out_frame_idx, out_obj_ids, out_mask_logits in predictor.add_new_points(inference_state):
        #     out_logits[find_frame(batch_no,out_frame_idx,BATCH_SIZE)]=out_mask_logits[0].cpu().numpy().tolist()
    out_logits["status"]="success"
    return  JSONResponse(content={"status": "success", **out_logits}, media_type="application/json")

def get_mask(mask, obj_id=None, random_color=False):
    color = np.ones(1)
    # h, w = mask.shape[-2:]
    mask_image = mask[0].astype(np.uint8) * color
    return mask_image

@app.post("/get_final_video")
async def process(data: dict):
    torch.cuda.empty_cache()
    batches = {}
    ann_obj_id = 1  # for now
    video_id = data['video_id']
    batch_size = int(video_id.split("_")[-1])
    frame_data = data["frame_data"]
    for i in frame_data:
        batch_number, index_number = find_batch(int(i["frame"]), batch_size)
        i["index"] = index_number - 1
        if not batches.get(batch_number):
            batches[batch_number] = [i]
        else:
            batches[batch_number].append(i)

    video_dir = os.path.join(VIDEO_DIR, video_id, "img_batch")
    if not os.path.exists(VIDEO_DIR):
        return JSONResponse(content={"status": "failed", "status_msg": "id does not exist"})

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid_dir = os.path.join(VIDEO_DIR, 'output2.mp4')
    print(out_vid_dir)
    # Load the first image of the first batch to get its size
    first_batch_dir = f"{video_dir}/batch_0/"
    first_image_path = os.path.join(first_batch_dir, sorted(os.listdir(first_batch_dir))[0])
    first_image = cv2.imread(first_image_path)
    height, width = first_image.shape[:2]

    out = cv2.VideoWriter(out_vid_dir, fourcc, 25, (width, height))
    for batch_no in range(1, len(os.listdir(video_dir)) + 1):
        video_batch_dir = f"{video_dir}/batch_{batch_no - 1}/"
        frame_data = batches.get(batch_no, [])

        if frame_data:
            inference_state = predictor.init_state(video_path=video_batch_dir)
            frame_data_dict = {}
            for i in frame_data:
                if i["frame"] in frame_data_dict:
                    frame_data_dict[i["frame"]].append(i)
                else:
                    frame_data_dict[i["frame"]] = [i]

            for _, i in frame_data_dict.items():
                cordinate = [[j["x"], j["y"]] for j in i]
                labels = np.array([1] * len(i), np.int32)
                points = np.array(cordinate, dtype=np.float32)
                _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                    inference_state=inference_state,
                    frame_idx=i[0]["index"],
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                )
            frame_names = [
                p for p in os.listdir(video_batch_dir)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

            for frame_name in os.listdir(video_batch_dir):
                frame_idx = int(frame_name.split('.')[0])
                if frame_idx in video_segments:
                    img = cv2.imread(os.path.join(video_batch_dir, frame_name))
                    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    converted_img = converted_img.astype(np.uint8)
                    _, out_mask = list(video_segments[frame_idx].items())[0]
                    out_mask[0].astype(np.uint8)
                    mask_img = get_mask(out_mask, random_color=True)
                    mask_img *= 255
                    mask_img = mask_img.astype(np.uint8)
                    mask_img = cv2.bitwise_not(mask_img)
                    result_img = cv2.bitwise_and(converted_img, converted_img, mask=mask_img)
                    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    out.write(result_img)
                else:
                    img = cv2.imread(os.path.join(video_batch_dir, frame_name))
                    out.write(img)
        else:
            for frame_name in os.listdir(video_batch_dir):
                print(frame_name)
                img = cv2.imread(os.path.join(video_batch_dir, frame_name))
                out.write(img)
    out.release()
    # Instead of returning a JSON response, return the video file
    return FileResponse(out_vid_dir, media_type="video/mp4", filename="output.mp4")
uvicorn.run(app, host="0.0.0.0", port=8000)
