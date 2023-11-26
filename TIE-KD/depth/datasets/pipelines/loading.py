import json
import mmcv
import numpy as np
import os.path as osp
from PIL import Image
from ..builder import PIPELINES
import sys
import matplotlib.pylab as plt


import torch
import torch.nn.functional as F
import torch.nn as nn

@PIPELINES.register_module()
class LoadKITTICamIntrinsic(object):
    """Load KITTI intrinsic
    """
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        # raw input
        if 'input' in  results['img_prefix']:
            date = results['filename'].split('/')[-5]
            results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]            
        # benchmark test
        else:
            temp = results['filename'].replace('benchmark_test', 'benchmark_test_cam')
            cam_file = temp.replace('png', 'txt')
            results['cam_intrinsic'] = np.loadtxt(cam_file).reshape(3, 3).tolist()
        # print("I;m LoadImage")
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class DepthLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']
        
        
        depth_gt = np.asarray(Image.open(filename),
                              dtype=np.float32) / results['depth_scale']                
        
        # print(np.max(depth_gt))
        # print(np.max(depth_gt)/256)
        # print("@@@@@@@@@@@@@@@@@@@@@")
        

        results['depth_gt'] = depth_gt
        results['depth_ori_shape'] = depth_gt.shape
        results['depth_fields'].append('depth_gt')

        # print("results['depth_ori_shape']")
        # print(results['depth_ori_shape'])
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

@PIPELINES.register_module()
class KDLoadAnnotations(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        #######################
        file_split = results['filename'].split('/')
        file_path = '{}_{}_{}_{}_{}'.format(file_split[3],file_split[4],file_split[5],file_split[6],file_split[7])
        kd_file = osp.join(file_path)
        
        # print(results['filename'])
        # print(results['kd_prefix'])
        # print(file_path)
        # sys.exit()

        if results.get('kd_prefix', None) is not None:
            filename = osp.join(results['kd_prefix'],
                                file_path)
        else:
            filename = file_path        
        # file_split = results['filename'].split('/')
        
        # if(file_split[1]=='cityscapes'):
        #     file_path = '{}_{}_{}'.format(file_split[3],file_split[4],file_split[5])
        # else:
        #     file_path = '{}_{}_{}_{}_{}'.format(file_split[3],file_split[4],file_split[5],file_split[6],file_split[7])
        # kd_file = osp.join(file_path)
        
        # if results.get('depth_prefix', None) is not None:
        #     filename = osp.join(results['depth_prefix'],
        #                         results['ann_info']['depth_map'])
        # else:
        #     filename = results['ann_info']['depth_map']
        ################
        # if results.get('kd_prefix', None) is not None:
        #     filename = osp.join(results['kd_prefix'],
        #                         file_path)
        # else:
        #     filename = file_path        

        #######################
        # if results.get('kd_prefix', None) is not None:
        #     filename = osp.join(results['kd_prefix'],
        #                         results['ann_info']['depth_map'])
        # else:
        #     filename = results['ann_info']['depth_map']
        #######################

        kd_gt = np.asarray(Image.open(filename),
                              dtype=np.float32)

        # print(np.max(kd_gt))
        # print(np.min(kd_gt))
        # print("@@@@@@@@@@@@@")
        # sys.exit()

        kd_gt = kd_gt /65535*80

        # test_image = kd_gt/80*255      
        # test_image = kd_gt              
        # test_image = test_image.astype('uint8')
        # kd_img = Image.fromarray(test_image)
        # kd_img.save('kd_gt.png')
        # sys.exit()

        

        # kd_gt = np.asarray(Image.open(filename), dtype=np.float32)*80/65535
        # kd_gt = 80 - kd_gt

        # kd_gt = np.asarray(Image.open(filename),
                            #   dtype=np.float32)/256
                

        
        # kd_gt_im= (kd_gt * 820).astype(np.uint16)
        # print(kd_gt_im.shape)
        # print(kd_gt_im[370][620])
        # for i in range(100):
            # kd_gt_im[370][600+i] = 0
            # print(kd_gt_im[10][600+i])

        # im =Image.fromarray(kd_gt_im)
        # im.save('/workspace/src/Monocular-Depth-Estimation-Toolbox/inference_result/kd_gt.png')
        # print(filename)
        # sys.exit()

        results['kd_gt'] = kd_gt
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class KDLoadAnnotations_NYU(object):
    """Load annotations for depth estimation.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        #######################
        file_split = results['filename'].split('/')   
        file_path = '{}_{}_{}_{}'.format(file_split[0], file_split[1], file_split[2], file_split[3])
        file_path = '{}{}'.format(file_path.split('.')[0],'.png')
        

        if results.get('kd_prefix', None) is not None:
            filename = osp.join(results['kd_prefix'],
                                file_path)
        else:
            filename = file_path        
        kd_gt = np.asarray(Image.open(filename),
                              dtype=np.float32)

        kd_gt = kd_gt/65535*10
        results['kd_gt'] = kd_gt        

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str


@PIPELINES.register_module()
class DisparityLoadAnnotations(object):
    """Load annotations for depth estimation.
    It's only for the cityscape dataset. TODO: more general.

    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """
    def __init__(self,
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='pillow'):
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`depth.CustomDataset`.

        Returns:
            dict: The dict contains loaded depth estimation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('depth_prefix', None) is not None:
            filename = osp.join(results['depth_prefix'],
                                results['ann_info']['depth_map'])
        else:
            filename = results['ann_info']['depth_map']

        if results.get('camera_prefix', None) is not None:
            camera_filename = osp.join(results['camera_prefix'],
                                       results['cam_info']['cam_info'])
        else:
            camera_filename = results['cam_info']['cam_info']

        with open(camera_filename) as f:
            camera = json.load(f)
        baseline = camera['extrinsic']['baseline']
        focal_length = camera['intrinsic']['fx']

        disparity = (np.asarray(Image.open(filename), dtype=np.float32) -
                     1.) / results['depth_scale']
        NaN = disparity <= 0

        disparity[NaN] = 1
        depth_map = baseline * focal_length / disparity
        depth_map[NaN] = 0

        results['depth_gt'] = depth_map
        results['depth_ori_shape'] = depth_map.shape

        results['depth_fields'].append('depth_gt')
        # print("it's disparity")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str



@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes,
                               flag=self.color_type,
                               backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)        

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                     dtype=np.float32),
                                       std=np.ones(num_channels,
                                                   dtype=np.float32),
                                       to_rgb=False)
        # print("It's load img")
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str




##until 230202
# import json
# import mmcv
# import numpy as np
# import os.path as osp
# from PIL import Image
# from ..builder import PIPELINES
# import sys
# import matplotlib.pylab as plt


# @PIPELINES.register_module()
# class LoadKITTICamIntrinsic(object):
#     """Load KITTI intrinsic
#     """
#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         # raw input
#         if 'input' in  results['img_prefix']:
#             date = results['filename'].split('/')[-5]
#             results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]            
#         # benchmark test
#         else:
#             temp = results['filename'].replace('benchmark_test', 'benchmark_test_cam')
#             cam_file = temp.replace('png', 'txt')
#             results['cam_intrinsic'] = np.loadtxt(cam_file).reshape(3, 3).tolist()
#         # print("I;m LoadImage")
#         return results


#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str


# @PIPELINES.register_module()
# class DepthLoadAnnotations(object):
#     """Load annotations for depth estimation.

#     Args:
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#     def __init__(self,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

    
#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('depth_prefix', None) is not None:
#             filename = osp.join(results['depth_prefix'],
#                                 results['ann_info']['depth_map'])
#         else:
#             filename = results['ann_info']['depth_map']
        
        
#         depth_gt = np.asarray(Image.open(filename),
#                               dtype=np.float32) / results['depth_scale']                
#         # print(depth_gt.shape)

#         results['depth_gt'] = depth_gt
#         results['depth_ori_shape'] = depth_gt.shape
#         results['depth_fields'].append('depth_gt')
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str

# @PIPELINES.register_module()
# class KDLoadAnnotations(object):
#     """Load annotations for depth estimation.

#     Args:
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#     def __init__(self,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         #######################
#         file_split = results['filename'].split('/')
#         file_path = '{}_{}_{}_{}_{}'.format(file_split[3],file_split[4],file_split[5],file_split[6],file_split[7])
#         kd_file = osp.join(file_path)

#         if results.get('kd_prefix', None) is not None:
#             filename = osp.join(results['kd_prefix'],
#                                 file_path)
#         else:
#             filename = file_path    
#         kd_gt = np.asarray(Image.open(filename),
#                               dtype=np.float32)
#         kd_gt = kd_gt *80/65535
        
#         # print(kd_gt)
#         # print(np.max(kd_gt))
#         # print(np.min(kd_gt))
#         # kd_gt = kd_gt *80/65535
         
#         #######################
#         # if results.get('kd_prefix', None) is not None:
#         #     filename = osp.join(results['kd_prefix'],
#         #                         results['ann_info']['depth_map'])
#         # else:
#         #     filename = results['ann_info']['depth_map']

#         # kd_gt = np.asarray(Image.open(filename),
#         #                       dtype=np.float32)*80/65535
#         #######################


        
#         # print(kd_gt)
#         # print(np.max(kd_gt))
#         # print(np.min(kd_gt))
#         # sys.exit()
        
#         # kd_gt = 80 - kd_gt

#         # kd_gt = np.asarray(Image.open(filename),
#                             #   dtype=np.float32)/256
                
#         # sys.exit()
        
#         # print(kd_gt.shape)

#         # print("=================")
#         # print(np.max(kd_gt))
#         # print(np.min(kd_gt))

#         # print(filename)
#         # print(kd_gt)
#         # print(np.max(kd_gt))
#         # sys.exit()
#         # print(np.min(kd_gt))
        
#         # kd_gt_im= (kd_gt * 820).astype(np.uint16)
#         # print(kd_gt_im.shape)
#         # print(kd_gt_im[370][620])
#         # for i in range(100):
#             # kd_gt_im[370][600+i] = 0
#             # print(kd_gt_im[10][600+i])

#         # im =Image.fromarray(kd_gt_im)
#         # im.save('/workspace/src/Monocular-Depth-Estimation-Toolbox/inference_result/kd_gt.png')
#         # print(filename)
#         # sys.exit()

#         kd_gt = np.expand_dims(kd_gt,axis=0)
#         results['kd_gt'] = kd_gt


#         # print("!---------")
#         # print(np.max(results['kd_gt']))
#         # print(np.min(results['kd_gt']))
#         # print(results['kd_gt'])
#         # sys.exit()
#         # print(filename)
#         # print(results['kd_prefix'])
#         # print(results['ann_info']['depth_map'])
#         # print(kd_gt.shape)
#         # results['depth_ori_shape'] = depth_gt.shape
#         # results['depth_fields'].append('depth_gt')
#         # print("it's depth anno")
#         # sys.exit()
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str



# @PIPELINES.register_module()
# class DisparityLoadAnnotations(object):
#     """Load annotations for depth estimation.
#     It's only for the cityscape dataset. TODO: more general.

#     Args:
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#     def __init__(self,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('depth_prefix', None) is not None:
#             filename = osp.join(results['depth_prefix'],
#                                 results['ann_info']['depth_map'])
#         else:
#             filename = results['ann_info']['depth_map']

#         if results.get('camera_prefix', None) is not None:
#             camera_filename = osp.join(results['camera_prefix'],
#                                        results['cam_info']['cam_info'])
#         else:
#             camera_filename = results['cam_info']['cam_info']

#         with open(camera_filename) as f:
#             camera = json.load(f)
#         baseline = camera['extrinsic']['baseline']
#         focal_length = camera['intrinsic']['fx']

#         disparity = (np.asarray(Image.open(filename), dtype=np.float32) -
#                      1.) / results['depth_scale']
#         NaN = disparity <= 0

#         disparity[NaN] = 1
#         depth_map = baseline * focal_length / disparity
#         depth_map[NaN] = 0

#         results['depth_gt'] = depth_map
#         results['depth_ori_shape'] = depth_map.shape

#         results['depth_fields'].append('depth_gt')
#         # print("it's disparity")
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str



# @PIPELINES.register_module()
# class LoadImageFromFile(object):
#     """Load an image from file.

#     Required keys are "img_prefix" and "img_info" (a dict that must contain the
#     key "filename"). Added or updated keys are "filename", "img", "img_shape",
#     "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
#     "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

#     Args:
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
#             Defaults to 'color'.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'cv2'
#     """
#     def __init__(self,
#                  to_float32=False,
#                  color_type='color',
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='cv2'):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call functions to load image and get image meta information.

#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('img_prefix') is not None:
#             filename = osp.join(results['img_prefix'],
#                                 results['img_info']['filename'])
#         else:
#             filename = results['img_info']['filename']
#         img_bytes = self.file_client.get(filename)
#         img = mmcv.imfrombytes(img_bytes,
#                                flag=self.color_type,
#                                backend=self.imdecode_backend)
#         if self.to_float32:
#             img = img.astype(np.float32)        

#         results['filename'] = filename
#         results['ori_filename'] = results['img_info']['filename']
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         # Set initial values for default meta_keys
#         results['pad_shape'] = img.shape
#         results['scale_factor'] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
#                                                      dtype=np.float32),
#                                        std=np.ones(num_channels,
#                                                    dtype=np.float32),
#                                        to_rgb=False)
#         # print("It's load img")
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32},'
#         repr_str += f"color_type='{self.color_type}',"
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str



# import json
# import mmcv
# import numpy as np
# import os.path as osp
# from PIL import Image
# from ..builder import PIPELINES


# @PIPELINES.register_module()
# class LoadKITTICamIntrinsic(object):
#     """Load KITTI intrinsic
#     """
#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         # raw input
#         if 'input' in  results['img_prefix']:
#             date = results['filename'].split('/')[-5]
#             results['cam_intrinsic'] = results['cam_intrinsic_dict'][date]
#         # benchmark test
#         else:
#             temp = results['filename'].replace('benchmark_test', 'benchmark_test_cam')
#             cam_file = temp.replace('png', 'txt')
#             results['cam_intrinsic'] = np.loadtxt(cam_file).reshape(3, 3).tolist()
        
#         return results


#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str


# @PIPELINES.register_module()
# class DepthLoadAnnotations(object):
#     """Load annotations for depth estimation.

#     Args:
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#     def __init__(self,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('depth_prefix', None) is not None:
#             filename = osp.join(results['depth_prefix'],
#                                 results['ann_info']['depth_map'])
#         else:
#             filename = results['ann_info']['depth_map']

#         depth_gt = np.asarray(Image.open(filename),
#                               dtype=np.float32) / results['depth_scale']

#         results['depth_gt'] = depth_gt
#         results['depth_ori_shape'] = depth_gt.shape

#         results['depth_fields'].append('depth_gt')
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str


# @PIPELINES.register_module()
# class DisparityLoadAnnotations(object):
#     """Load annotations for depth estimation.
#     It's only for the cityscape dataset. TODO: more general.

#     Args:
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'pillow'
#     """
#     def __init__(self,
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='pillow'):
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call function to load multiple types annotations.

#         Args:
#             results (dict): Result dict from :obj:`depth.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded depth estimation annotations.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('depth_prefix', None) is not None:
#             filename = osp.join(results['depth_prefix'],
#                                 results['ann_info']['depth_map'])
#         else:
#             filename = results['ann_info']['depth_map']

#         if results.get('camera_prefix', None) is not None:
#             camera_filename = osp.join(results['camera_prefix'],
#                                        results['cam_info']['cam_info'])
#         else:
#             camera_filename = results['cam_info']['cam_info']

#         with open(camera_filename) as f:
#             camera = json.load(f)
#         baseline = camera['extrinsic']['baseline']
#         focal_length = camera['intrinsic']['fx']

#         disparity = (np.asarray(Image.open(filename), dtype=np.float32) -
#                      1.) / results['depth_scale']
#         NaN = disparity <= 0

#         disparity[NaN] = 1
#         depth_map = baseline * focal_length / disparity
#         depth_map[NaN] = 0

#         results['depth_gt'] = depth_map
#         results['depth_ori_shape'] = depth_map.shape

#         results['depth_fields'].append('depth_gt')
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str


# @PIPELINES.register_module()
# class LoadImageFromFile(object):
#     """Load an image from file.

#     Required keys are "img_prefix" and "img_info" (a dict that must contain the
#     key "filename"). Added or updated keys are "filename", "img", "img_shape",
#     "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
#     "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

#     Args:
#         to_float32 (bool): Whether to convert the loaded image to a float32
#             numpy array. If set to False, the loaded image is an uint8 array.
#             Defaults to False.
#         color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
#             Defaults to 'color'.
#         file_client_args (dict): Arguments to instantiate a FileClient.
#             See :class:`mmcv.fileio.FileClient` for details.
#             Defaults to ``dict(backend='disk')``.
#         imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
#             'cv2'
#     """
#     def __init__(self,
#                  to_float32=False,
#                  color_type='color',
#                  file_client_args=dict(backend='disk'),
#                  imdecode_backend='cv2'):
#         self.to_float32 = to_float32
#         self.color_type = color_type
#         self.file_client_args = file_client_args.copy()
#         self.file_client = None
#         self.imdecode_backend = imdecode_backend

#     def __call__(self, results):
#         """Call functions to load image and get image meta information.

#         Args:
#             results (dict): Result dict from :obj:`mmseg.CustomDataset`.

#         Returns:
#             dict: The dict contains loaded image and meta information.
#         """

#         if self.file_client is None:
#             self.file_client = mmcv.FileClient(**self.file_client_args)

#         if results.get('img_prefix') is not None:
#             filename = osp.join(results['img_prefix'],
#                                 results['img_info']['filename'])
#         else:
#             filename = results['img_info']['filename']
#         img_bytes = self.file_client.get(filename)
#         img = mmcv.imfrombytes(img_bytes,
#                                flag=self.color_type,
#                                backend=self.imdecode_backend)
#         if self.to_float32:
#             img = img.astype(np.float32)

#         results['filename'] = filename
#         results['ori_filename'] = results['img_info']['filename']
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['ori_shape'] = img.shape
#         # Set initial values for default meta_keys
#         results['pad_shape'] = img.shape
#         results['scale_factor'] = 1.0
#         num_channels = 1 if len(img.shape) < 3 else img.shape[2]
#         results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
#                                                      dtype=np.float32),
#                                        std=np.ones(num_channels,
#                                                    dtype=np.float32),
#                                        to_rgb=False)
#         return results

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += f'(to_float32={self.to_float32},'
#         repr_str += f"color_type='{self.color_type}',"
#         repr_str += f"imdecode_backend='{self.imdecode_backend}')"
#         return repr_str
