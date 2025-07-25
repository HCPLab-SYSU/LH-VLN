import torch
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,0'
import numpy as np
from .eva_clip import create_model_and_transforms
from .load import load_img
# from sklearn.metrics.pairwise import euclidean_distances

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load visual encoder
# device = torch.device('cuda:1')

model, _, transform = create_model_and_transforms("EVA02-CLIP-L-14-336", "/data2/songxinshuai/nav_gen/data/models/EVA02_CLIP_L_336_psz14_s6B.pt", force_custom_clip=True)
visual_encoder = model.visual.to(device)
visual_encoder.eval()

def get_image_embedding(images):
    # vision_x = [transform(image).unsqueeze(0).to(args.device) for image in images]
    vision_x = [transform(image).unsqueeze(0).to(device) for image in images]
    vision_x = torch.cat(vision_x, dim=0)
    with torch.no_grad():
        outs = visual_encoder.forward_features(vision_x)
    outs = outs.data.cpu().numpy()
    # outs = outs.data.mean(dim=0).cpu().numpy()
    # print(outs.shape)
    return outs

# file = '/data2/songxinshuai/nav_gen/dataset/dataa'
# image_embeddings = []
# for task in os.listdir(file):
#     file_name = file + "/" + task
#     # trials = os.listdir(file_name)
#     # trial_file = file_name+ "/" + trials[0] + '/raw_images'
#     images = load_img(file_name)
#     image_embedding = get_image_embedding(images)
#     image_embeddings.append(image_embedding)

# # 计算平均欧几里得距离
# avg_euclidean_distance = np.mean(euclidean_distances(image_embeddings))

# print("Average Euclidean distance: ", avg_euclidean_distance)