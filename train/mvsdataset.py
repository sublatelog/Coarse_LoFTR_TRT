from torch.utils.data import Dataset
import torch
import itertools
import os
from pathlib import Path
from numpy.random import default_rng
import cv2
import numpy as np
import re

from utils import make_query_image, ratio_preserving_resize


# Dataset parsing routines were taken from https://github.com/YoYo000/MVSNet/


def get_view_pairs(file_name, image_files, cams_files, depth_files):
    view_pairs = []
    with open(file_name) as file:
        lines = file.readlines() # 
        for line in lines[1:]:
            if len(line) > 3:
                tokens = line.split() # 10 10 2346.41 1 2036.53 9 1243.89 12 1052.87 11 1000.84 13 703.583 2 604.456 8 439.759 14 327.419 27 249.278 
                pair_files = []
                
                # 同じ対象のIDを調べる
                for token in tokens[1::2]:
                    img_id = token.zfill(3) # 文字列を特定の文字数になるように0で埋める
#                     img_id = token.zfill(8) # 文字列を特定の文字数になるように0で埋める
                    
                    for img_file_name, cam_file_name, depth_file_name in zip(image_files, cams_files, depth_files):
                        text_name = str(img_file_name) # rect_001_5_r5000.png
                        
                        # ファイル名にIDを含み、maskでない場合、(img_file_name, cam_file_name, depth_file_name)のファイルセットとして追加する。                       
                        if img_id in text_name and 'mask' not in text_name:
                            pair_files.append((img_file_name, cam_file_name, depth_file_name))
                            
                # 同じ対象のファイルセットのリストから長さ2の順列組み合わせを取り出す。            
                pairs = itertools.permutations(pair_files, r=2) 
                
                view_pairs.extend(pairs)
    return view_pairs


class DataCamera:
    def __init__(self):
        self.extrinsic = np.zeros((4, 4), dtype=np.float)
        self.intrinsic = np.zeros((3, 3), dtype=np.float)
        self.depth_min = 0
        self.depth_interval = 0
        self.depth_num = 0
        self.depth_max = 0

    def get_dir(self):
        r = self.get_rot_matrix()
        r_inv = np.linalg.inv(r)
        z = np.array([0, 0, 1, 1])
        dir = z.dot(r_inv.T)
        return dir[:3]

    # 外部パラメーターの位置部を取り出す
    def get_pos(self):
        t = self.extrinsic[:, 3]
        return t

    def get_pos_inv(self):
        r = self.get_rot_matrix()
        r_inv = np.linalg.inv(r)
        t = self.extrinsic[:, 3]
        camera_pos = t.dot(r_inv.T)
        camera_pos[:3] *= -1
        return camera_pos[:3]

    # 単位行列に外部パラメーターを上書き
    def get_rot_matrix(self):
        r = np.eye(4) # 単位行列
        """
        array([[1., 0., 0., 0.],
               [0., 1., 0., 0.],
               [0., 0., 1., 0.],
               [0., 0., 0., 1.]])
       """
        
        r[0:3, 0:3] = self.extrinsic[0:3, 0:3]
        return r

    # 3Dの位置情報の操作して2D位置を作成
    def project_points(self, coordinates_3d):
        coordinates_cam = coordinates_3d.dot(self.extrinsic.T) # 3D位置と外部パラメーターの内積
        coordinates_cam = coordinates_cam / coordinates_cam[:, [3]] # 3行目で割る

        intrinsic_ex = np.pad(self.intrinsic, ((0, 0), (0, 1)), 'constant', constant_values=((0, 0), (0, 0))) # 内部パラメーターをパディングで調節
        coordinates_2d = coordinates_cam.dot(intrinsic_ex.T) # 3D位置と内部パラメーターの内積で2Dに変える
        coordinates_2d = coordinates_2d / coordinates_2d[:, [2]] # 2行目で割る
        return coordinates_2d, coordinates_cam[:, [2]] # 2D位置, 3D位置

    # 2Dの位置情報の操作
    def back_project_points(self, coordinates_2d, depth):
        """
        self : png
        coordinates_2d : [row, col]
        depth : depth_hw1[coordinates_2d:[col], coordinates_2d:[row], np.newaxis]
        """
        
        # from pixel to camera space
        """
        self.intrinsic
        [[361.54125   0.       82.9005 ]
         [  0.      360.39625  66.38375]
         [  0.        0.        1.     ]]
        """
        intrinsic_inv = np.linalg.inv(self.intrinsic) # np.zeros((3, 3)), np.linalg.inv():逆行列を求める
        """
        intrinsic_inv
        [[ 0.00276594  0.         -0.22929852]
         [ 0.          0.00277471 -0.18419592]
         [ 0.          0.          1.        ]]
        """
        
        """
        coordinates_2d
        [[  0   0   1]
         [  0  16   1]
         [  0  32   1]
         [  0  48   1]
         [  0  64   1]
         [  0  80   1]
         [  0  96   1]
         [  0 112   1]
         [ 16   0   1]
         [ 16  16   1]
         [ 16  32   1]
         [ 16  48   1]
        """
        
        """
        depth
        [[  0.     ]
         [  0.     ]
         [700.183  ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [832.4816 ]
        """
        
        coordinates_2d = coordinates_2d * depth
        """
        coordinates_2d
        [[    0.             0.             0.        ]
         [    0.             0.             0.        ]

         [    0.             0.             0.        ]
         [22405.85546875 78420.49414062   700.1829834 ]
         [    0.             0.             0.        ]
         [    0.             0.             0.        ]
         [    0.             0.             0.        ]
         [    0.             0.             0.        ]
         [    0.             0.             0.        ]
         [    0.             0.             0.        ]
         [39959.11816406 79918.23632812   832.48162842]
         [33391.74609375 77914.07421875   695.66137695]
         [    0.             0.             0.        ]
        """
        
        coordinates_cam = coordinates_2d.dot(intrinsic_inv.T)  # [x, y, z]
        """
        coordinates_cam
        [[  0.           0.           0.        ]
         [  0.           0.           0.        ]
         [-98.57751776  88.62359483 700.1829834 ]
         [  0.           0.           0.        ]
         [  0.           0.           0.        ]
         [  0.           0.           0.        ]
         [  0.           0.           0.        ]
         [  0.           0.           0.        ]
         [  0.           0.           0.        ]
         [-80.36215285  68.41053012 832.48162842]
         [-67.15445002  88.05128583 695.66137695]
         [  0.           0.           0.        ]
         [  0.           0.           0.        ]
        """
        
        # make homogeneous
        coordinates_cam = np.hstack([coordinates_cam, np.ones_like(coordinates_cam[:, [0]])])

        # from camera to world space
        r = self.get_rot_matrix() # 単位行列にextrinsicを上書き
        """
        r
        [[-0.636298  -0.727666   0.25618    0.       ]
         [ 0.0315712  0.307237   0.951109   0.       ]
         [-0.770797   0.613276  -0.172521   0.       ]
         [ 0.         0.         0.         1.       ]]
        """
        
        r_inv = np.linalg.inv(r) # 逆行列
        """
        r_inv
        [[ 0.65705269 -0.01296394 -0.75373308 -0.        ]
         [-0.65305156  0.48967028 -0.57770798 -0.        ]
         [ 0.37656991  0.87181177  0.31327303  0.        ]
         [ 0.          0.          0.          1.        ]]
        """
        
        t = self.extrinsic[:, 3] # 外部パラメーターの位置部分
        """
        t
        [-271.044 -494.295  361.453    1.   ]
        self.intrinsic
        [[361.54125    0.        82.900375]
         [  0.       360.3975    66.38375 ]
         [  0.         0.         1.      ]]
        """

        coordinates_cam[:, :3] -= t[:3]
        coordinates_world = coordinates_cam.dot(r_inv.T) # dot():内積
        """
        coordinates_world
        [[ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
         [ 512.63883005 -392.41179956  718.94893333    1.        ]
        """

        return coordinates_world


def load_camera_matrices(file_name):
#     print(file_name)
    with open(file_name) as file:
        camera = DataCamera()
        words = file.read().split()
        
#         print(len(words))
        assert (len(words) == 29) # 31
        
        """
        extrinsic 4x4　外部パラメーター
        4:-0.0956172 -0.903896 0.416929 -268.918
        4:0.247582 0.38409 0.889482 -549.69
        4:-0.964137 0.188274 0.187063 539.553
        4:0.0 0.0 0.0 1.0

        intrinsic 3x3 内部パラメーター
        3:2892.33 0 823.205
        3:0 2883.18 619.069
        3:0 0 1

        2:425 2.5
        """
        
        for i in range(0, 4): # 0,1,2,3
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1  # 1,2,3,4,, 5,6,7,8,, 9,10,11,12,, 13,14,15,16,,
                camera.extrinsic[i][j] = words[extrinsic_index]

        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18  # 18,19,20,, 21,22,23,, 24,25,26,,
                camera.intrinsic[i][j] = words[intrinsic_index]

        camera.depth_min = float(words[27])
        camera.depth_interval = float(words[28])
#         camera.depth_num = int(float(words[29]))
#         camera.depth_max = float(words[30])
        return camera


def load_pfm(file_name):
    with open(file_name, mode='rb') as file:
        header = file.readline().decode('UTF-8').rstrip()
        
        """
        Pf
        160 128
        -1.000000
        """

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
            
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups()) # 160 128
        else:
            raise Exception('Malformed PFM header.')
            
        scale = float((file.readline()).decode('UTF-8').rstrip()) # -1.000000
        
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
            
        # データ部の取得
        data_string = file.read()
        data = np.fromstring(data_string, data_type) # 文字列で書かれた行列・配列を数値に変換
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = cv2.flip(data, 0)
        return data


class MVSDataset(Dataset):
    def __init__(self, path, image_size, resolution, depth_tolerance=0.005, seed=0, epoch_size=0,
                 return_cams_info=False):
        self.path = path # /content/datasets/mvs_training/dtu
        self.image_size = image_size
        self.items = []
        self.epoch_size = epoch_size
        self.resolution = resolution # (16, 4)[0]
        self.return_cams_info = return_cams_info
        self.depth_tolerance = depth_tolerance

#         mvs_folders = list(Path(self.path).glob('*'))
        num_train = 1
#         num_train = 128
        for i in range(num_train):
#         for folder_name in mvs_folders:
            folder_name = self.path
            images_folder = os.path.join(folder_name, 'Rectified/scan1_train')
#             images_folder = os.path.join(folder_name, 'blended_images')
            image_files = list(Path(images_folder).glob('rect_[0-9][0-9][0-9]_0_r5000.*')) # rect_001_5_r5000.png
#             image_files = list(Path(images_folder).glob('*[0-9].*'))
            image_files.sort()
    
            print("image_files:", len(image_files))

            cams_folder = os.path.join(folder_name, 'Cameras/train')
#             cams_folder = os.path.join(folder_name, 'cams')
            cams_files = list(Path(cams_folder).glob('*cam.*'))
            cams_files.sort()
    
            print("cams_files:", len(cams_files))

            depth_folder = os.path.join(folder_name, 'Depths/scan1_train')
#             depth_folder = os.path.join(folder_name, 'rendered_depth_maps')
            depth_files = list(Path(depth_folder).glob('depth_map_*.*'))
#             depth_files = list(Path(depth_folder).glob('*.*'))
            depth_files.sort()
    
            print("depth_files:", len(depth_files))

            pairs_file = os.path.join(folder_name, 'Cameras', 'pair.txt')
#             pairs_file = os.path.join(folder_name, 'cams', 'pair.txt')
            if os.path.exists(pairs_file):
                
                # 同じ対象のファイルセットのリストから長さ2の順列組み合わせを取り出す。
                view_pairs = get_view_pairs(pairs_file, image_files, cams_files, depth_files)
                self.items.extend(view_pairs)

        print("len:", len(self.items))
        
        # シャッフル
        self.rng = default_rng(seed)       
        self.rng.shuffle(self.items)
        
        # epoch_sizeで切る
        if epoch_size != 0:
            self.epoch_items = self.items[:epoch_size]

    def reset_epoch(self):
        self.rng.shuffle(self.items)
        if self.epoch_size != 0:
            self.epoch_items = self.items[:self.epoch_size]

    def __getitem__(self, index):
        # 同じ対象のファイルセットの長さ2の順列組み合わせからindex指定で取り出す。
        (img_file_name1, cam_file_name1, depth_file_name1), (img_file_name2, cam_file_name2, depth_file_name2) = self.items[index]
        
        img1 = cv2.imread(str(img_file_name1))
        
        # size
        img_size_orig = np.array([img1.shape[1], img1.shape[0]])
        
        # img1のサイズに調節
        img1 = make_query_image(img1, self.image_size)
        
        # img1のサイズに調節
        img2 = cv2.imread(str(img_file_name2))
        img2 = make_query_image(img2, self.image_size)

        img1 = torch.from_numpy(img1)[None] / 255.0
        img2 = torch.from_numpy(img2)[None] / 255.0

        
        conf_matrix, camera1, camera2 = self.generate_groundtruth_confidence(cam_file_name1,
                                                                             depth_file_name1,
                                                                             cam_file_name2,
                                                                             depth_file_name2)
        conf_matrix = torch.from_numpy(conf_matrix)[None]

        if self.return_cams_info:
            return img1, img2, conf_matrix, img_size_orig, camera1.intrinsic, camera1.get_rot_matrix(), camera1.get_pos_inv(), camera2.intrinsic, camera2.get_rot_matrix(), camera2.get_pos_inv()
        else:
            return img1, img2, conf_matrix

    def generate_groundtruth_confidence(self, cam_file_name1, depth_file_name1, cam_file_name2, depth_file_name2):
        data_camera1 = load_camera_matrices(cam_file_name1)
        data_camera2 = load_camera_matrices(cam_file_name2)

        depth_hw1 = load_pfm(depth_file_name1)
        depth_hw2 = load_pfm(depth_file_name2)
        
        """
        [[  0.        0.        0.      ...   0.        0.        0.     ]
         [  0.        0.        0.      ...   0.        0.        0.     ]
         [  0.        0.        0.      ...   0.        0.        0.     ]
         ...
         [637.763   637.63104 637.8569  ... 608.58246 608.456   608.52106]
         [634.9358  633.941   633.69403 ... 606.6304  606.5178  606.60925]
         [630.75995 630.63495 630.03937 ... 605.18945 605.22174 605.23083]]
        """

        original_image_size = depth_hw1.shape # pf_size:160 128

        w = original_image_size[1] // self.resolution # (16, 4)[0], 128 / 16 = 8
        h = original_image_size[0] // self.resolution # 160/16=10

        # (w, h)で型を作る
        coordinates_2d = np.array(list(np.ndindex(w, h))) * self.resolution # [[0-7]:[0-9]]*16
        """
        [[  0   0]
         [  0  16]
         [  0  32]
         [  0  48]
         [  0  64]
         [  0  80]
         [  0  96]
         [  0 112]
         [ 16   0]
         [ 16  16]
         [ 16  32]
        """
        coordinates_2d = np.hstack([coordinates_2d, np.ones_like(coordinates_2d[:, [0]])]) # 値が1の列を追加
        """
        [[  0   0   1]
         [  0  16   1]
         [  0  32   1]
         [  0  48   1]
         [  0  64   1]
         [  0  80   1]
         [  0  96   1]
         [  0 112   1]
         [ 16   0   1]
         [ 16  16   1]
         [ 16  32   1]
         [ 16  48   1]
         [ 16  64   1]
         [ 16  80   1]
         [ 16  96   1]
         [ 16 112   1]
         [ 32   0   1]
        """
        # depth_hw1から型をcoordinates_2dの(行, 列, ？)で抜き出す
        depth1 = depth_hw1[coordinates_2d[:, 1], coordinates_2d[:, 0], np.newaxis]
        """
        [[  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [  0.     ]
         [593.61505]
         [597.86456]
         [602.0165 ]
         [606.6527 ]
         [612.2571 ]
        """
        # 型とdepthでdata_camera1から3D位置を作成
        coordinates1_3d = data_camera1.back_project_points(coordinates_2d, depth1)
        """
        [[ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
         [ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
         [ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
         [ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
         [ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
         [ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
         [ 4.83172032e+02  2.78608916e+02  3.15340735e+02  1.00000000e+00]
        """
        
        # 3D位置から2D位置を作成
        coordinates2, depth2_computed = data_camera2.project_points(coordinates1_3d)
        """
        coordinates2
        [[-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [-436.96860492  422.54865705    1.        ]
         [  37.68580728   -4.11082869    1.        ]
         [  38.24905828    6.11399674    1.        ]
        """
        
        """
        depth2_computed
        [[ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [ 30.35929586]
         [600.00655943]
         [609.97034427]
         [619.92222389]
        """
        # check depth consistency
        coordinates2_clipped = np.around(coordinates2) # 小数点以下の四捨五入
        
        mask = np.where(
                        np.all( # 全要素が条件を満たす
                            (coordinates2_clipped[:, :2] >= (0, 0)) & (coordinates2_clipped[:, :2] < (original_image_size[1], original_image_size[0])), 
                            axis=1
                        ))
        """
        mask
        (array([10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28,
               29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47,
               53, 54, 55, 61, 62, 63, 69, 70, 71, 78]),)
        """
        
        coordinates2_clipped = coordinates2_clipped[mask].astype(np.long)
        coordinates2 = coordinates2[mask]
        coordinates_2d = coordinates_2d[mask]
        depth2_computed = depth2_computed[mask]
        """
        depth2_computed
        [[553.84071301]
         [558.60513158]
         [563.16237646]
         [568.4362655 ]
         [573.91236636]
         [579.25411086]
         [592.019649  ]
         [547.56295082]
         [551.59414192]
         [555.72251929]
         [560.37811786]
         [565.18635127]
         [570.16694413]
         """

        depth2 = depth_hw2[coordinates2_clipped[:, 1], coordinates2_clipped[:, 0], np.newaxis]
        
        depth2[depth2 == 0.0] = np.finfo(depth2.dtype).max # np.finfo:「数値型データ」が取り得る値の範囲
        
        # 差の割合が許容範囲に収まっているかチェック
        depth_consistency_mask, _ = np.where(np.absolute((depth2 - depth2_computed) / depth2) < self.depth_tolerance)

        coordinates2 = coordinates2[depth_consistency_mask]
        coordinates_2d = coordinates_2d[depth_consistency_mask]

        # filter image coordinates to satisfy the grid property
        region_threshold = self.resolution / 3  # pixels
        
        grid_mask = np.where(
                            np.all(
                                    (self.resolution - coordinates2[:, :2] % self.resolution) <= region_threshold, 
                                    axis=1
                            )
        )
        
        coordinates2 = coordinates2[grid_mask]
        coordinates_2d = coordinates_2d[grid_mask]

        # scale coordinates to the training size
        scale_w = self.image_size[0] / original_image_size[1]
        scale_h = self.image_size[1] / original_image_size[0]

        # scale the first image coordinates
        coordinates1 = coordinates_2d.astype(np.float)
        coordinates1[:, :2] *= np.array([scale_w, scale_h])
        coordinates1[:, :2] /= self.resolution
        coordinates1 = np.around(coordinates1)

        # scale the second image coordinates
        coordinates2[:, :2] *= np.array([scale_w, scale_h])
        coordinates2[:, :2] /= self.resolution
        coordinates2 = np.around(coordinates2)

        # check bounds correctness
        w = self.image_size[0] // self.resolution
        h = self.image_size[1] // self.resolution

        mask = np.where(
                        np.all(
                            (coordinates2[:, :2] >= (0, 0)) & (coordinates2[:, :2] < (w, h)),
                            axis=1
                        ))
        coordinates2 = coordinates2[mask][:, :2]
        coordinates1 = coordinates1[mask][:, :2]

        # fill the confidence matrix
        coordinates1 = coordinates1[:, 1] * w + coordinates1[:, 0]
        coordinates2 = coordinates2[:, 1] * w + coordinates2[:, 0]

        conf_matrix = np.zeros((h * w, h * w), dtype=float)
        
        # coordinates1とcoordinates2の両方を満たす箇所を1.0にする（他は0.0）
        conf_matrix[coordinates1.astype(np.long), coordinates2.astype(np.long)] = 1.0
        """
        conf_matrix
        [[0. 0. 0. ... 0. 0. 0.]
         [0. 0. 0. ... 0. 0. 0.]
         [0. 0. 0. ... 0. 0. 0.]
         ...
         [0. 0. 0. ... 0. 0. 0.]
         [0. 0. 0. ... 0. 0. 0.]
         [0. 0. 0. ... 0. 0. 0.]]
        """
        
        return conf_matrix, data_camera1, data_camera2

    def __len__(self):
        if self.epoch_size != 0:
            return self.epoch_size
        return len(self.items)
