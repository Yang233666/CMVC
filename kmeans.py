import torch
import numpy as np
import random
import sys

device_gpu = torch.device('cuda')
device_cpu = torch.device('cpu')

# Choosing `num_centers` random data points as the initial centers
def random_init(dataset, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    used = torch.zeros(num_points, dtype=torch.long)
    indices = torch.zeros(num_centers, dtype=torch.long)
    for i in range(num_centers):
        while True:
            cur_id = random.randint(0, num_points - 1)
            if used[cur_id] > 0:
                continue
            used[cur_id] = 1
            indices[i] = cur_id
            break
    indices = indices.to(device_gpu)
    centers = torch.gather(dataset, 0, indices.view(-1, 1).expand(-1, dimension))
    return centers

# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)
    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long, device=device_gpu)
    centers_t = torch.transpose(centers, 0, 1)
    # print('centers:', type(centers), centers.shape, centers)
    centers_norms = torch.sum(centers ** 2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece ** 2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        # print('centers_t:', type(centers_t), centers_t.shape, centers_t)
        # print('distances:', type(distances), distances.shape, distances)
        distances *= -2.0
        # print('distances:', type(distances), distances.shape, distances)
        distances += dataset_norms
        # print('dataset_norms:', type(dataset_norms), dataset_norms.shape, dataset_norms)
        # print('distances:', type(distances), distances.shape, distances)
        distances += centers_norms
        # print('centers_norms:', type(centers_norms), centers_norms.shape, centers_norms)
        # print('distances:', type(distances), distances.shape, distances)
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
        # print('codes:', type(codes), codes.shape, codes)
        # exit()
    return codes, distances

# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float, device=device_gpu)
    cnt = torch.zeros(num_centers, dtype=torch.float, device=device_gpu)
    centers.scatter_add_(0, codes.view(-1, 1).expand(-1, dimension), dataset)
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float, device=device_gpu))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float, device=device_gpu))
    centers /= cnt.view(-1, 1)
    return centers

def cluster(dataset, num_centers):
    centers = random_init(dataset, num_centers)
    codes, distances = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        # sys.stdout.write('.')
        sys.stdout.flush()
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers)
        new_codes, new_distances = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            # sys.stdout.write('\n')
            # print('Converged in %d iterations' % num_iterations)
            break
        codes = new_codes
        distances = new_distances
    return centers, codes, distances


def max_margin_loss(positive_score, negative_score):
    gamma = 0.1 * negative_score
    score = positive_score - negative_score + gamma
    # print('gamma:', gamma)
    # print('positive_score:', positive_score)
    # print('negative_score:', negative_score)
    # print('score:', score)
    if score > 0:
        loss = score
    else:
        loss = score - score
    # print('loss:', loss)
    # exit()
    return loss

def positive_loss(positive_score):
    score = positive_score
    if score > 0:
        loss = score
    else:
        loss = score - score
    return loss

def kmeans_train(dataset, num_centers):
    # print('Starting clustering')
    centers, codes_in, distances_in = cluster(dataset, num_centers)

    d_in = 0
    for i, center in enumerate(codes_in):
        d_in += distances_in[i][center]
    # d_in = d_in / len(codes_in)
    # print('d_in:', d_in)
    codes_out, distances_out = compute_codes(centers, centers)

    d_out = 0
    d_out += sum(sum(distances_out))
    for i in range(len(distances_out)):
        d_out -= distances_out[i][i]
    d_out = d_out / (len(distances_out[0]) - 1) / 2
    # print('d_out:', d_out)
    # exit()

    return d_in, d_out
