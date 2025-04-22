# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 11:37:23 2025

@author: almeida-roch
"""

import os
import cv2
import numpy as np
from skimage import io, filters, morphology, measure
from skimage.transform import resize
from skimage.morphology import ball, closing, remove_small_objects
from matplotlib import pyplot as plt
import trimesh
import imageio

import time
start_time = time.time()

#PART 1: BINARIZATION AND 3D MATRIX GENERATION

input_folder = 'MicroCT'
output_folder_bin = 'MicroCTbi'
os.makedirs(output_folder_bin, exist_ok=True)

roi_size = 300
roi_position = [0, 0, roi_size, roi_size]

image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])


print('Begin image processing...')

#Lista para as imagens binarizadas
binarized_stack = []

for i, fname in enumerate(image_files):
    print(f'processing {i+1}/{len(image_files)}')
    img_path = os.path.join(input_folder, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    x, y, w, h = roi_position
    cropped = img[y:y+h, x:x+w]
    resized = cv2.resize(cropped, (roi_size, roi_size))

    # Otsu
    thresh_val = filters.threshold_otsu(resized)
    binary = resized > thresh_val

    binarized_stack.append(binary.astype(np.uint8))

binarized_images = np.stack(binarized_stack, axis=-1)
np.save(os.path.join(output_folder_bin, 'binarized_images.npy'), binarized_images)

print('Final of binarization')

# Salvar slices para BoneJ
output_slices_folder = 'MicroCT_Pre2'
os.makedirs(output_slices_folder, exist_ok=True)

for i in range(binarized_images.shape[2]):
    img = (binarized_images[:, :, i] * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(output_slices_folder, f'slice_{i:03}.tif'), img)



binarized_closed = binarized_images.copy()
binarized_closed[:, :, 0] = 0      # Fecha a base
binarized_closed[:, :, -1] = 0     # Fecha o topo
binarized_closed[0, :, :] = 0      # Fecha lado esquerdo
binarized_closed[-1, :, :] = 0     # Fecha lado direito
binarized_closed[:, 0, :] = 0      # Fecha face frontal
binarized_closed[:, -1, :] = 0     # Fecha face traseira

verts, faces, _, _ = measure.marching_cubes(binarized_closed, 0.5)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.export('sample_pre.stl')

#PARTE 2: REAÇÃO-DIFUSÃO (Gray-Scott)

U = np.ones_like(binarized_images, dtype=np.float32)
V = np.zeros_like(binarized_images, dtype=np.float32)

V[binarized_images == 1] = 1
U[binarized_images == 1] = 0.5

Du, Dv = 0.0001, 0.005
F, k = 0.02, 0.005
dt = 1.0
steps = 50

def laplacian(X):
    return (
        np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) +
        np.roll(X, 1, axis=1) + np.roll(X, -1, axis=1) +
        np.roll(X, 1, axis=2) + np.roll(X, -1, axis=2) -
        6 * X
    )

for t in range(steps):
    Lu = laplacian(U)
    Lv = laplacian(V)
    reaction = U * V**2
    U += (Du * Lu - reaction + F * (1 - U)) * dt
    V += (Dv * Lv + reaction - (F + k) * V) * dt

# Filtros

processed = V > 0.15
processed = closing(processed, ball(2))
processed = remove_small_objects(processed, 150)

# Salva slices modificados para BoneJ
output_processed_folder = 'MicroCT_Proc2'
os.makedirs(output_processed_folder, exist_ok=True)

for i in range(processed.shape[2]):
    img = (processed[:, :, i] * 255).astype(np.uint8)
    imageio.imwrite(os.path.join(output_processed_folder, f'slice_{i:03}.tif'), img)



# Isosuperfície


processed_closed = processed.copy()
processed_closed[:, :, 0] = 0      # Fecha a base
processed_closed[:, :, -1] = 0     # Fecha o topo
processed_closed[0, :, :] = 0      # Fecha lado esquerdo
processed_closed[-1, :, :] = 0     # Fecha lado direito
processed_closed[:, 0, :] = 0     # Fecha face frontal
processed_closed[:, -1, :] = 0     # Fecha face traseira

verts_mod, faces_mod, _, _ = measure.marching_cubes(processed_closed, 0.5)
mesh_mod = trimesh.Trimesh(vertices=verts_mod, faces=faces_mod)
mesh_mod.export('sample_modified.stl')


# Tentando visualizar a isosuperfície aqui no Python

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                color='lightgray', lw=0.1, alpha=1.0)
ax.set_box_aspect([1,1,1])
plt.title("Original Isosurface")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts_mod[:, 0], verts_mod[:,1], faces_mod, verts_mod[:, 2],
                color='lightgray', lw=0.1, alpha=1.0)
ax.set_box_aspect([1,1,1])
plt.title("Modified Isosurface")
plt.tight_layout()
plt.show()

end_time = time.time()
print(f"Time: {end_time - start_time:.2f} segundos")
