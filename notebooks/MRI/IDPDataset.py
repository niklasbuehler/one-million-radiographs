# %%
import functools
import hashlib
import inspect
import itertools
import json
import os
import pickle
import re
import stat
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path
import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydicom
import seaborn as sns
import torch
import torch.utils.data
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from IPython import get_ipython
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
import skimage.io
from tqdm import tqdm

while not Path('.toplevel').exists() and Path('..').resolve() != Path().resolve():
    os.chdir(Path('..'))
if str(Path().resolve()) not in sys.path:
    sys.path.insert(0, str(Path().resolve()))


import simonlanger

pd.set_option('max_colwidth', 20)

# %%

#'fracture', 'foreignmaterial', 'bodypart_original',


class IDPDatasetBase(torch.utils.data.Dataset):
    def __init__(self, size=224, max_size_padoutside=None, annotated=False, fracturepseudolabled=False, basedir='/home/langers/langer_idp/dataset', required_cols='auto', cache=True, diskcache_reldir='../IDPDatasetBaseCache', diskcache_reldir_autoappend=True, return_df_row=False, return_custom_cols=(), normalization_mode=0.99, bodypartexamined_mappingloc='data/BodyPartExamined_mappings_mergemore.json', bodypartexamined_dropna=False, clean_brightedges=False, clean_rotation=False, merge_scapula_shoulder=False, no_pixelarray_loading=False):
        """
        normalization_mode (float|'max'|None): None means no normalization is applied (the conversion to a float32 tensor nevertheless takes place), float: output is a 0-1-clipped normalization where >= normalization_mode quantile is 1
        """

        super().__init__()
        self.basedir = Path(basedir)
        self.size = size
        # max_size_padoutside overrides size setting if not None
        self.max_size_padoutside = max_size_padoutside
        self.annotated = annotated
        self.fracturepseudolabled = fracturepseudolabled
        self.cache = cache
        self.diskcache_reldir = diskcache_reldir
        self.return_df_row = return_df_row
        self.return_custom_cols = return_custom_cols
        self.normalization_mode = normalization_mode
        self.clean_brightedges = clean_brightedges
        self.clean_rotation = clean_rotation
        self.no_pixelarray_loading = no_pixelarray_loading

        if self.clean_brightedges and (self.max_size_padoutside is not None):
            raise ValueError('clean_rotation requires max_size_padoutside=None')
        if self.clean_rotation and (not self.clean_brightedges):
            raise ValueError('clean_rotation requires clean_brightedges=True')
        if self.clean_brightedges and self.size != 224:
            print(
                f'[WARN] clean_brightedges with non-default image size {self.size} instead of {224} may not yield good results')

        if self.fracturepseudolabled and (not self.annotated):
            raise ValueError('fracturepseudolabled requires annotated=True')

        if required_cols == 'auto':
            if not self.annotated:
                required_cols = ('path', 'bodypart', 'patientid', 'examinationid', 'findingspath', 'findings',
                                 'dcm_SOPInstanceUID', 'dcm_PatientID', 'dcm_BodyPartExamined', 'pixelarr_shape')
            elif not self.fracturepseudolabled:
                required_cols = ('path', 'bodypart', 'fracture', 'foreignmaterial', 'bodypart_original', 'annotated',
                                 'patientid', 'examinationid', 'findingspath', 'findings', 'dcm_SOPInstanceUID', 'dcm_PatientID', 'dcm_BodyPartExamined', 'pixelarr_shape')
            else:
                required_cols = ('path', 'bodypart', 'fracture', 'foreignmaterial', 'bodypart_original', 'annotated',
                                 'patientid', 'examinationid', 'findingspath', 'findings', 'dcm_SOPInstanceUID', 'dcm_PatientID', 'dcm_BodyPartExamined', 'pixelarr_shape',
                                 'fracturenum', 'fracture_pseudolabel', 'fracture_bestlabel', 'fracture_bestlabeltext')

        print('initializing IDPDatasetBase ...')
        if self.diskcache_reldir is not None:
            self.diskcache_reldir = Path(self.diskcache_reldir)
            if diskcache_reldir_autoappend:
                self.diskcache_reldir = self.diskcache_reldir.with_name(self.diskcache_reldir.name +
                                                                        ('_cleanbrightedges' if self.clean_brightedges else '') +
                                                                        ('_cleanrotation' if self.clean_rotation else '') + 
                                                                        (f'_{self.size}' if self.max_size_padoutside is None and self.size != 224 else ''))
            if not (self.basedir / self.diskcache_reldir).exists():  # if is symlink crashes with just the mkdir(exist_ok=True line)
                (self.basedir / self.diskcache_reldir).mkdir(exist_ok=True)
            self._getpixelarray_load_funcstrsha256 = hashlib.sha256(
                inspect.getsource(self._getpixelarray_load).encode('utf8')).hexdigest()

        def getdf(mayuseslow=True):
            dataset_thin_loc = Path(
                f'data/cache/dataset_thin{"_annotated" if self.annotated else ""}{"_fracturepseudolabled" if self.fracturepseudolabled else ""}.pt')
            dataset_full_loc = Path(
                f'data/cache/dataset{"_annotated" if self.annotated else ""}{"_fracturepseudolabled" if self.fracturepseudolabled else ""}.pkl')

            def getdf_slow():
                assert mayuseslow
                print(f'reading {dataset_full_loc} file' +
                      (f' and re-generating {dataset_thin_loc} ' if required_cols is not None else '') + ' ...')
                df_full = pd.read_pickle(dataset_full_loc)
                if required_cols is None:
                    return df_full

                df = df_full.copy()
                df = df[list(required_cols)]
                torch.save((required_cols, df), dataset_thin_loc)
                return getdf(mayuseslow=False)

            if required_cols is None or not dataset_thin_loc.exists():
                return getdf_slow()

            read_requiredcols, df = torch.load(dataset_thin_loc)
            if read_requiredcols != required_cols:
                return getdf_slow()

            if mayuseslow:
                print(f'fast initialization from {dataset_thin_loc}')
            return df

        df_full = getdf()
        df = df_full.copy()

        if bodypartexamined_mappingloc is not None:
            bodypartexamined_mapping = json.loads(Path(bodypartexamined_mappingloc).read_text())
            df['dcm_BodyPartExamined_str'] = [(seq if not seq != seq else '').lower()
                                              for seq in df['dcm_BodyPartExamined']]

            df['mapped_BodyPartExamined'] = [bodypartexamined_mapping[BodyPartExamined_str.replace(' ', '').lower()]
                                             for BodyPartExamined_str in df['dcm_BodyPartExamined_str']]

            if bodypartexamined_dropna:
                df = df.dropna(subset=['mapped_BodyPartExamined'])

        df_labelcomparison_loc = Path('data/cache/df_labelcomparison.pkl')
        if df_labelcomparison_loc.exists():
            df_labelcomparison = pd.read_pickle()

            df_full['dcm_BodyPartExamined_str'] = [(seq if not seq != seq else '').lower()
                                                   for seq in df_full['dcm_BodyPartExamined']]

            filtermask = (df_labelcomparison.fillna(
                0) - df_labelcomparison.fillna(0).astype(int)) > 0.1
            filter_rowcol_tuples = list(
                zip(*np.nonzero(filtermask.to_numpy())))

            for row, col in filter_rowcol_tuples:
                dcmbodypart = list(df_labelcomparison.index)[row]
                bodypart = list(df_labelcomparison.columns)[col][1]

                df = df[~((df['dcm_BodyPartExamined_str'] == dcmbodypart)
                          & (df['bodypart'] == bodypart))]
            print(f'{len(df_full)-len(df)=} items excluded by df_labelcomparison.pkl')
        else:
            print(
                f'{df_labelcomparison_loc} does not exit --> no items excluded by it')

        df['relpathstr'] = simonlanger.ensureunique(
            [os.path.relpath(path, basedir) for path in df['path']])
        df = df.sort_values('relpathstr')
        df = df.reset_index(drop=True)
        self.bodypart_to_idx = {
            bodypart: i for i, bodypart in enumerate(sorted(set(df['bodypart'])))
        }
        self.idx_to_bodypart = {
            i: bodypart for i, bodypart in enumerate(sorted(set(df['bodypart'])))
        }

        self.fracturepseudolables = ['0_no', '1_yes', '2_votesfew', '3_contradiction', '4_unsure']
        self.fracture_to_idx = {
            bodypart: i for i, bodypart in enumerate(self.fracturepseudolables[:2])
        }

        if merge_scapula_shoulder:
            df['bodypart'] = df['bodypart'].replace('scapula', 'schulter')
            print('replaced all scapula bodyparts with shoulder')

        self.df = df.copy()
        print(self, 'initialized')

# -%%

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'IDPDatasetBase(len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    @functools.cache
    def _getitem_innercached(self, index):
        return self._getitem_inner(index)

    def _getpixelarray_load(self, curitem_series):

        dcm = pydicom.dcmread(curitem_series['path'])
        pixel_array = dcm.pixel_array
        # normalize by max
        match (self.normalization_mode):
            case None:
                pass
            case 'max':
                pixel_array /= pixel_array.max()
            case num:
                pixel_array = pixel_array.astype(float)
                pixel_array /= np.quantile(pixel_array, self.normalization_mode)
                pixel_array = np.clip(pixel_array, 0, 1)

        # add batch dim
        pixel_array = torch.tensor(pixel_array, dtype=torch.float32)[None]

        if self.max_size_padoutside is not None:
            pixel_array = TF.resize(pixel_array, size=self.max_size_padoutside - 1, max_size=self.max_size_padoutside)

            missing_cols = self.max_size_padoutside - pixel_array.shape[-1]
            pad_left = missing_cols // 2
            pad_right = sum(divmod(missing_cols, 2))

            missing_rows = self.max_size_padoutside - pixel_array.shape[-2]
            pad_top = missing_rows // 2
            pad_bottom = sum(divmod(missing_rows, 2))

            pixel_array = F.pad(pixel_array, [pad_left, pad_right, pad_top, pad_bottom])
        else:
            # typical case for augmented trainings, allow for varying image sizes from here
            if self.size is not None:
                pixel_array = TF.resize(pixel_array, self.size)

            if self.clean_brightedges:
                def _clean(pixel_array):
                    pixel_array = np.array(pixel_array)
                    img_8bit = cv2.normalize(pixel_array, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
                    _, thresh = cv2.threshold(img_8bit, 0, 1, cv2.THRESH_BINARY)

                    contours, hierarchy = cv2.findContours(
                        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                    page = sorted(contours, key=cv2.contourArea, reverse=True)[:1]  # has to be largest area match

                    # draw contours on the original image
                    contours_only = np.zeros_like(pixel_array)
                    cv2.drawContours(image=contours_only, contours=page,
                                     contourIdx=-1, color=1., thickness=12, lineType=cv2.LINE_AA)

                    if len(page) > 0:
                        # Loop over the contours.
                        c = page[0]
                        # Approximate the contour.
                        epsilon = 0.02 * cv2.arcLength(c, True)
                        corners = cv2.approxPolyDP(c, epsilon, True)
                        # If our approximated contour has four points

                    if len(page) == 0 or len(corners) != 4:
                        # print('[WARN] failed to find 4 corners; fallback to outer border cropping')
                        corners = [
                            [0, 0],
                            [0, pixel_array.shape[0] - 1],
                            [pixel_array.shape[1] - 1, pixel_array.shape[0] - 1],
                            [pixel_array.shape[1] - 1, 0],
                        ]
                        corners = np.array(corners)[:, None, :]
                        contours_only = np.zeros_like(pixel_array)
                        cv2.drawContours(image=contours_only, contours=[corners],
                                         contourIdx=-1, color=1., thickness=12, lineType=cv2.LINE_AA)

                        # raise NotImplementedError('TODO')

                    # import time
                    # starttime = time.time()
                    img_masked = pixel_array * (contours_only == 0) + pixel_array * 0.0 * (contours_only != 0)
                    img_blurred = cv2.GaussianBlur(img_masked, (35, 35), 5.)
                    contours_only = cv2.GaussianBlur(contours_only, (15, 15), 0)
                    # endtime = time.time()
                    # print(endtime - starttime)
                    img_combined = pixel_array * (1 - contours_only) + img_blurred * (contours_only)

                    # plt.imshow(img_combined, cmap='bone')
                    # plt.show()
                    if not self.clean_rotation:
                        return img_combined

                    corners = sorted(np.concatenate(corners).tolist())
                    corners = order_points(corners)
                    corners = np.array(corners)
                    # Calculate the angle of rotation needed to align the image with the x-axis
                    angle_calc = np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]) * 180 / np.pi

                    # Calculate the angle of rotation needed to align the image with the x-axis
                    angle_x = np.arctan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0]) * 180 / np.pi

                    # Calculate the angle of rotation needed to align the image with the y-axis
                    angle_y = np.arctan2(corners[2, 0] - corners[1, 0], corners[2, 1] - corners[1, 1]) * 180 / np.pi
                    angle_y *= -1

                    angle_calc = sorted([angle_x, angle_y], key=abs)[0]

                    angle = angle_calc if -45 < angle_calc < 45 else 0
                    rotation_matrix = cv2.getRotationMatrix2D((img_8bit.shape[1] / 2, img_8bit.shape[0] / 2), angle, 1)
                    img_combined_rotated = cv2.warpAffine(img_combined, rotation_matrix, img_combined.shape[::-1])

                    return img_combined_rotated

                pixel_array = _clean(pixel_array[0, :, :])
                # re-add batch dim
                pixel_array = torch.tensor(pixel_array, dtype=torch.float32)[None]
        return pixel_array

    def _getpixelarray(self, index, curitem_series):
        if self.diskcache_reldir is not None:
            s = os.stat(curitem_series['path'])
            cur_equalconfig_dict = {
                'index': index,
                # 'curitem_series': curitem_series,
                'path': curitem_series['path'],
                'path_stat': {k: getattr(s, k) for k in dir(s) if k.startswith('st_') and not k.startswith('st_atime')},
                'self.normalization_mode': self.normalization_mode,
                'self.max_size_padoutside': self.max_size_padoutside,
                'self.size': self.size,
                'self.clean_brightedges': self.clean_brightedges,
                'self.clean_rotation': self.clean_rotation,
                'self._getpixelarray_load_funcstrsha256': self._getpixelarray_load_funcstrsha256,
            }

            # try to read from cache first
            cacheloc = self.basedir / self.diskcache_reldir / f'{index}.pt'
            if cacheloc.exists():
                for _ in range(4):
                    try:
                        equalconfig_dict, pixel_array = torch.load(cacheloc)
                        if cur_equalconfig_dict == equalconfig_dict:
                            return pixel_array
                        break
                    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
                        print('[warn] Error encountered while reading cached image file', index, e)
                        time.sleep(5)
                        pass
                simonlanger.print_ratelimited('[warn] unable to read or outdated cached image file')

        pixel_array = self._getpixelarray_load(curitem_series)

        if self.diskcache_reldir is not None:
            try:
                torch.save((cur_equalconfig_dict, pixel_array), cacheloc)
            except RuntimeError as e:
                print('[warn] RuntimeError encountered while writing cached image file', index, e)
        return pixel_array

    def _getitem_inner(self, index):
        curitem_series = self.df.loc[index]
        pixel_array = self._getpixelarray(index, curitem_series) if not self.no_pixelarray_loading else None
        res = dict(pixel_array=pixel_array, bodypart_idx=self.bodypart_to_idx[curitem_series['bodypart']])
        if self.annotated:
            res['fracture'] = curitem_series['fracture']
        if self.fracturepseudolabled:
            res['fracture_bestlabel'] = curitem_series['fracture_bestlabel']
        if self.return_df_row:
            res['row'] = curitem_series

        for customcol in self.return_custom_cols:
            res[customcol] = curitem_series[customcol]
        return res

    def __getitem__(self, index):
        if self.cache:
            return self._getitem_innercached(index)
        else:
            return self._getitem_inner(index)


def order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    # https://learnopencv.com/automatic-document-scanner-using-opencv/
    # https://medium.com/analytics-vidhya/opencv-findcontours-detailed-guide-692ee19eeb18

    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # Return the ordered coordinates.
    return rect.astype('int').tolist()
#

# %%


class IDPDataset(torch.utils.data.Dataset):
    def __init__(self, dsbase, mode, stratification_target='bodypart', seed=42, val_size=0.2, test_size=0.2, extra_filter=None):
        super().__init__()
        self.dsbase = dsbase
        self.mode = mode
        print(f'\ninitializing IDPDataset(mode={self.mode}) ...')

        if not mode in ['train', 'val', 'test', 'train+val', 'train+val+test']:
            raise ValueError('invalid IDPDataset mode')
        modeset = set(mode.split('+'))
        self.modeset = modeset

        stratification_target_frequencies = dsbase.df[stratification_target].value_counts()
        # if multiple stratification target values for the same patient, use the rarest one for stratification
        # this computation is always performed globally

        self.df = dsbase.df.copy()
        self.df.reset_index(names='dsbase_index', inplace=True)

        split_test_loc = Path(f'data/split_test_straton_{stratification_target}.csv')
        if not split_test_loc.exists():
            res = input(
                f'WARN: NO TRAINVAL TEST SPLIT FOUND AT {split_test_loc}, type YESGENERATE[enter] to generate one: ')
            if res.strip() != 'YESGENERATE':
                self.df = None
                exit(1)

            print('WARN: GENERATING NEW TRAINVAL TEST SPLIT')
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.df.groupby('dcm_PatientID')}

            _, test = train_test_split(list(patientid_to_strattarget.keys()),
                                       test_size=test_size, stratify=list(patientid_to_strattarget.values()), random_state=0)
            test_patientids = pd.DataFrame(test)
            test_patientids.to_csv(split_test_loc)
        test_patientids = pd.read_csv(split_test_loc, index_col=0)

        patientid_index_df = self.df.set_index('dcm_PatientID')
        assert set(patientid_index_df.index).issuperset(test_patientids['0'])
        test_idxs = patientid_index_df.loc[test_patientids['0']]['dsbase_index']
        if 'test' in modeset:
            print('WARN: including test data')
            if modeset == {'test'}:
                # remove trainval
                self.df = self.df.loc[test_idxs]
                assert len(set(self.df['dcm_PatientID']) - set(test_patientids['0'])) == 0
            assert len(set(self.df['dcm_PatientID']) & set(test_patientids['0'])) == len(test_patientids)
        else:
            self.df = self.df.drop(test_idxs)
            assert len(set(self.df['dcm_PatientID']) & set(test_patientids['0'])) == 0

        patientid_index_df = self.df.set_index('dcm_PatientID')
        if ('train' in modeset or 'val' in modeset) and not ('train' in modeset and 'val' in modeset):
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in self.df.groupby('dcm_PatientID')}

            train, val = train_test_split(list(patientid_to_strattarget.keys()),
                                          test_size=val_size, stratify=list(patientid_to_strattarget.values()), random_state=seed)
            train_patientids = pd.DataFrame(train).rename(columns={0: '0'})
            val_patientids = pd.DataFrame(val).rename(columns={0: '0'})
            val_idxs = patientid_index_df.loc[val_patientids['0']]['dsbase_index']
            if 'val' in modeset:
                # since not both, only keep the val ones
                self.df = self.df.loc[val_idxs]
                assert len(set(self.df['dcm_PatientID']) & set(val_patientids['0'])) == len(val_patientids)
                assert len(set(self.df['dcm_PatientID']) - set(val_patientids['0'])) == 0
            else:
                self.df = self.df.drop(val_idxs)
                assert len(set(self.df['dcm_PatientID']) & set(train_patientids['0'])) == len(train_patientids)
                assert len(set(self.df['dcm_PatientID']) & set(val_patientids['0'])) == 0

        if extra_filter is not None:
            self.df = extra_filter(self.df)
        self.df = self.df.copy()
        print(self, 'initialized')

# -%%

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'IDPDataset(mode={self.mode}, len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    def __getitem__(self, index):
        return self.dsbase[self.df.iloc[index]['dsbase_index']]

    def getrow(self, index):
        return self.df.iloc[index]


class IDPDatasetFracture(torch.utils.data.Dataset):
    """
        Performs a specialized data split on IDPDatasetBase (argument dsbase):

        first, images with the human fracture label Unsure are removed from any further consideration
        second, ALL images patients for which at least one human-labled image exists are excluded from consideration for train

        all images in val/test are exclusively human-labled 'yes'/'no'
        images in train may have either a pseudo-label or no ('yes'/'no') label at all, if no label at all may occur is controlled by train_include_nopseudolabel

        test_size specifies which fraction of the exclusively human-labled images shall be used for test, the remainder will be val
    """

    def __init__(self, dsbase, mode, train_include_nopseudolabel=None, seed=42, test_size=0.5, extra_filter=None):
        super().__init__()
        self.dsbase = dsbase
        self.mode = mode
        self.train_include_nopseudolabel = train_include_nopseudolabel
        print(
            f'\ninitializing IDPDatasetFracture(mode={self.mode}, train_include_nopseudolabel={self.train_include_nopseudolabel}) ...')

        if not mode in ['train', 'val', 'test', 'train+val', 'train+val+test']:
            raise ValueError('invalid IDPDataset mode')
        modeset = set(mode.split('+'))
        self.modeset = modeset

        if 'train' in modeset and self.train_include_nopseudolabel is None:
            raise ValueError('train_include_nopseudolabel may not be none if training data is to be included')

        self.df = dsbase.df.copy()
        self.df.reset_index(names='dsbase_index', inplace=True)

        self.df = self.df[self.df['fracture'] != 'Unsure']
        stratification_target = 'stratification_bodypart__fracture_bestlabeltext'
        self.df[stratification_target] = self.df['bodypart'] + \
            '__' + self.df['fracture_bestlabeltext']
        self.df[stratification_target] = self.df[stratification_target].str.replace(
            'hws__1_yes', 'hws__0_no', regex=False)  # only a single hws__1_yes instance exists, so treat it in stratification like hws__0_no

        # reliable evaluation source
        df_reliableeval = self.df[self.df['annotated']]

        if 'train' in modeset:
            # conditions:
            # patientid must not be present val/test
            # no annotations available (redundant given 1st condition, but kept for sake of readability)
            # fracture_pseudolabel <= 1 means only no=0 and yes=1 are permitted, no error conditions such as too few votes (2) or contradition (3)

            mask_patientid_notin_reliableval = ~self.df['dcm_PatientID'].isin(set(df_reliableeval['dcm_PatientID']))
            mask_nonannoated = ~self.df['annotated']

            # verifies redundancy claim: A implied B <==> ~A | B ; here, we have ~A and ~B given mask_nonann.../mask_patientid...
            assert ((mask_nonannoated) | (~mask_patientid_notin_reliableval)).all()

            df_traindata = self.df[
                mask_patientid_notin_reliableval &
                mask_nonannoated &
                (self.train_include_nopseudolabel | (self.df['fracture_pseudolabel'] <= 1))
            ]

        reliableeval_stratification_target_frequencies = df_reliableeval[stratification_target].value_counts()
        # if multiple stratification target values for the same patient, use the rarest one for stratification
        # this computation is always performed globally

        split_test_loc = Path(f'data/split_test_straton_{stratification_target}.csv')
        if not split_test_loc.exists():
            res = input(
                f'WARN: NO TEST SPLIT FOUND AT {split_test_loc}, type YESGENERATE[enter] to generate one: ')
            if res.strip() != 'YESGENERATE':
                self.df = None
                exit(1)

            print('WARN: GENERATING NEW TEST SPLIT')
            patientid_to_strattarget = {patientid: sorted(set(subdf[stratification_target]),
                                                          key=lambda x: reliableeval_stratification_target_frequencies.loc[x])[0]
                                        for patientid, subdf in df_reliableeval.groupby('dcm_PatientID')}

            _, test = train_test_split(list(patientid_to_strattarget.keys()),
                                       test_size=test_size, stratify=list(patientid_to_strattarget.values()), random_state=0)
            test_patientids = pd.DataFrame(test)
            test_patientids.to_csv(split_test_loc)
        test_patientids = pd.read_csv(split_test_loc, index_col=0)

        ##
        output_dfs = []

        patientid_index_df = df_reliableeval.set_index('dcm_PatientID')
        assert set(patientid_index_df.index).issuperset(test_patientids['0'])
        test_idxs = patientid_index_df.loc[test_patientids['0']]['dsbase_index']
        if 'test' in modeset:
            print('WARN: including test data')
            output_dfs.append(df_reliableeval.loc[test_idxs])

        if 'val' in modeset:
            output_dfs.append(df_reliableeval.drop(test_idxs))

        if 'train' in modeset:
            output_dfs.append(df_traindata)

        self.df = pd.concat(output_dfs)
        del output_dfs

        if 'test' not in modeset:
            assert len(set(self.df['dcm_PatientID']) & set(test_patientids['0'])) == 0
        if 'val' not in modeset:
            assert len(set(self.df['dcm_PatientID']) & set(df_reliableeval.drop(test_idxs)['dcm_PatientID'])) == 0

        if extra_filter is not None:
            self.df = extra_filter(self.df)
        self.df = self.df.copy()
        print(self, 'initialized')

# -%%

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        return f'IDPDatasetFracture(mode={self.mode}, train_include_nopseudolabel={self.train_include_nopseudolabel}, len={len(self.df)})'

    def __str__(self) -> str:
        return repr(self)

    def __getitem__(self, index):
        return self.dsbase[self.df.iloc[index]['dsbase_index']]

    def getrow(self, index):
        return self.df.iloc[index]


def labelcomparison(dsbase, indexcol='mapped_BodyPartExamined'):
    df = dsbase.df

    df_labelcomparison = pd.pivot_table(df, index=[indexcol], columns=['bodypart'], values=['patientid'],
                                        sort=False, aggfunc=lambda x: x.count())

    df_labelcomparison_loc = Path('data/df_labelcomparison.pkl')

    if df_labelcomparison_loc.exists():
        df_labelcomparison = pd.read_pickle(df_labelcomparison_loc)
        print('read existing labelcomparison state')

    def plotlyshowimgs(subdf):
        downsample_factor = 4
        nels = min(4**2, len(subdf))
        cols = int(np.ceil(np.sqrt(nels)))

        def matplotlib_to_plotly(cmap, pl_entries):
            h = 1.0 / (pl_entries - 1)
            pl_colorscale = []

            for k in range(pl_entries):
                C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
                pl_colorscale.append([k * h, 'rgb' + str((C[0], C[1], C[2]))])

            return pl_colorscale

        def get_pixels(path):
            try:
                return pydicom.dcmread(path).pixel_array
            except AttributeError:
                return np.ones((1, 1)) * np.nan

        pixelarrs = list(simonlanger.multitheaded_map(
            get_pixels, list(subdf['path'])[:nels], threads=0))
        combined_pixelarr = np.ones([nels,
                                    np.max([x.shape[0] for x in pixelarrs]),
                                    np.max([x.shape[1] for x in pixelarrs])]) * np.nan
        for i, pixelarr in enumerate(pixelarrs):
            if len(pixelarr.shape) == 3:
                print('[WARN]: unusual pixelarr with shape ', pixelarr.shape)
                pixelarr = pixelarr[:, :, 0]

            # normalize for visualization here by max value in img, suboptimal since removes physical meaning of values
            combined_pixelarr[i, :pixelarr.shape[0], :pixelarr.shape[1]] = pixelarr.astype(
                float) / np.max(pixelarr)

        combined_pixelarr = combined_pixelarr[:,
                                              ::downsample_factor,
                                              ::downsample_factor]
        print(combined_pixelarr.shape)

        fig = px.imshow(combined_pixelarr, facet_col=0, facet_col_wrap=cols,
                        color_continuous_scale=matplotlib_to_plotly(plt.cm.bone, 255))
        fig.show(renderer='browser')

    def on_pick(showfig, event):
        artist = event.artist
        xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
        # x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        # print('Artist picked:', event.artist)
        # print('{} vertices picked'.format(len(ind)))
        # print('Pick between vertices {} and {}'.format(min(ind), max(ind) + 1))
        # print('x, y of mouse: {:.2f},{:.2f}'.format(xmouse, ymouse))

        dcmbodypart, bodypart = list(df_labelcomparison.index)[int(
            ymouse)], list(df_labelcomparison.columns)[int(xmouse)][1]

        if 'right' in str(event.mouseevent.button).lower():
            curval = df_labelcomparison.iloc[int(ymouse), int(xmouse)]
            print(curval)
            if (curval - int(curval)) > 0.1:
                df_labelcomparison.iloc[int(ymouse), int(xmouse)] = int(curval)
            else:
                df_labelcomparison.iloc[int(ymouse), int(
                    xmouse)] = int(curval) + 0.5

            df_labelcomparison.to_pickle(df_labelcomparison_loc)
            showfig()
        else:
            subdf = df[(df[indexcol] == dcmbodypart)
                       & (df['bodypart'] == bodypart)]

            infostr = f'{dcmbodypart=}, {bodypart=}, {len(subdf)=}'
            print(infostr, event.mouseevent.button)
            plotlyshowimgs(subdf)

    def showfig():
        get_ipython().run_line_magic('matplotlib', 'qt')

        sns.set(rc={'figure.figsize': (15, 17)})
        heatmap = sns.heatmap(df_labelcomparison, annot=True, fmt='.1f',
                              cmap='Blues', picker=True, linecolor='black', linewidths=1)
        fig = heatmap.get_figure()
        fig.canvas.callbacks.connect(
            'pick_event', functools.partial(on_pick, showfig))
        mngr = plt.get_current_fig_manager()
        fig.canvas.manager.window.move(0, 0)
        # geom = mngr.window.geometry()
        # x,y,dx,dy = geom.getRect()
        mngr.window.showMaximized()
        fig.show()
        return fig

    return showfig()


def augment_transforms(output_size, args, dsbase, finetune, istrain=True):
    if args.lightly_imagecollate:
        from typing import Union, Tuple
        from lightly.transforms import GaussianBlur
        from lightly.data.collate import _random_rotation_transform

        cj_prob: float = 0.8
        cj_bright: float = 0.7
        cj_contrast: float = 0.7
        cj_sat: float = 0.7
        cj_hue: float = 0.2
        min_scale: float = 0.15
        random_gray_scale: float = 0.2
        gaussian_blur: float = 0.5
        kernel_size: float = 0.1
        vf_prob: float = 0.0
        hf_prob: float = 0.5
        rr_prob: float = 0.0
        rr_degrees: Union[None, float, Tuple[float, float]] = None
        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        transforms = [
            T.RandomResizedCrop(size=output_size, scale=(min_scale, 1.0)),
            _random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
        ]
        # transforms += [
        #     T.ToTensor(),
        # ]

        return T.Compose(transforms)

    def pad_to_largersize(pixel_array):
        padsize = max(pixel_array.shape[-2:])
        missing_cols = padsize - pixel_array.shape[-1]
        pad_left = missing_cols // 2
        pad_right = sum(divmod(missing_cols, 2))

        missing_rows = padsize - pixel_array.shape[-2]
        pad_top = missing_rows // 2
        pad_bottom = sum(divmod(missing_rows, 2))

        pixel_array = F.pad(pixel_array, [pad_left, pad_right, pad_top, pad_bottom])
        return pixel_array

    # transforms = torchvision.transforms.Compose(
    #     ([torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.75)] if istrain else []) +
    #     ([pad_to_largersize, ]) +
    #     ([torchvision.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.5, 1.1), shear=30), ] if istrain else []) +
    #     ([torchvision.transforms.Resize(output_size),])
    #     # torchvision.transforms.RandomResizedCrop(output_size),
    # )

    aug_gauge = getattr(args, 'aug_gauge', True)

    if not finetune:
        transforms = T.Compose(
            ([GaugeAugmentation(dsbase)] if istrain and aug_gauge else []) +
            ([T.ColorJitter(brightness=0.5, contrast=0.75)] if istrain else []) +
            ([] if istrain and args.aug_randomcrop else [pad_to_largersize]) +
            ([T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.5), shear=30) if args.aug_randomcrop else T.RandomAffine(degrees=45, translate=(0.3, 0.3), scale=(0.4, 1.5), shear=45)] if istrain else []) +
            ([T.RandomResizedCrop(output_size)] if istrain and args.aug_randomcrop else [T.Resize(output_size), ])
            # torchvision.transforms.RandomResizedCrop(output_size),
        )
    else:
        # if finetune
        transforms = T.Compose(
            ([GaugeAugmentation(dsbase)] if istrain and aug_gauge else []) +
            ([T.ColorJitter(brightness=0.1, contrast=0.2)] if istrain else []) +
            ([] if istrain and args.aug_randomcrop else [pad_to_largersize]) +
            ([T.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.1), shear=10) if args.aug_randomcrop else T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.1), shear=15)] if istrain else []) +
            ([T.RandomResizedCrop(output_size, scale=(0.95, 1), ratio=(0.9, 1.1))]
             if istrain and args.aug_randomcrop else [T.Resize(output_size), ])
            # torchvision.transforms.RandomResizedCrop(output_size),
        )

    return transforms


class GaugeAugmentation(torch.nn.Module):
    def __init__(self, dsbase,
                 numgauges_weights=(1., 1., 1.), scaling_bounds=(0.8, 1.2), opacity_bounds=(0.75, 1.), gaugechoice_weights=None,
                 gauge_directory='data/aug_gauge'):
        super().__init__()
        self.dsbase = dsbase
        self.gauges = GaugeAugmentation.get_gauges(dsbase, gauge_directory)
        self.numgauges_weights = torch.tensor(numgauges_weights, dtype=float)
        self.scaling_bounds = torch.tensor(scaling_bounds, dtype=float)
        self.opacity_bounds = torch.tensor(opacity_bounds, dtype=float)
        self.gaugechoice_weights = gaugechoice_weights
        if self.gaugechoice_weights is None:
            self.gaugechoice_weights = torch.ones(len(self.gauges), dtype=float)
        self.gaugechoice_weights = torch.tensor(self.gaugechoice_weights, dtype=float)

    @staticmethod
    @functools.cache
    def get_gauges(dsbase, directory):
        paths = sorted(Path(directory).glob('*.tiff'))
        target_imagesize = dsbase.size
        if target_imagesize is None:
            raise ValueError('dsbase.size must be set for this augmentation to function')

        result = []
        for path in paths:
            imgidx = int(path.with_suffix('').name)
            # TF.resize(size) scales SMALLER side to size
            img_side_scaledto_targetsize = min(dsbase.df.iloc[imgidx]['pixelarr_shape'])
            default_downscale_factor = target_imagesize / img_side_scaledto_targetsize

            gauge = torch.movedim(torch.tensor(skimage.io.imread(path)), -1, 0)
            gauge_minside = min(gauge.shape[1:])
            result.append([gauge, gauge_minside * default_downscale_factor])
            # # debug sanity check
            # el = dsbase[imgidx]
            # img = el['pixel_array']
            # downsized_gauge = TF.resize(gauge, int(gauge_minside * default_downscale_factor))
            # posy, posx = 0, 0
            # masked_gauge = torch.zeros_like(img)
            # masked_gauge[:, posy:posy + downsized_gauge.shape[1], posx:posx + downsized_gauge.shape[2]] = downsized_gauge[1] * downsized_gauge[0]

            # outimg = (1 - masked_gauge) * img + masked_gauge
            # plt.imshow(outimg[0], cmap='bone')
            # plt.show()
        return result

    def forward(self, img):
        numgauges, = torch.multinomial(self.numgauges_weights, 1)
        if numgauges == 0:
            return img
        gauge_idxs = torch.multinomial(self.gaugechoice_weights, numgauges)
        for gauge_idx in gauge_idxs:
            gauge, default_size = self.gauges[gauge_idx]
            downsized_gauge = TF.resize(gauge, int(default_size * np.random.uniform(*self.scaling_bounds)))
            opacity = np.random.uniform(*self.opacity_bounds)
            mask_gauge = torch.zeros_like(img)
            posy = np.random.randint(0, img.shape[1] - downsized_gauge.shape[1])
            posx = np.random.randint(0, img.shape[2] - downsized_gauge.shape[2])
            mask_gauge[:, posy:posy + downsized_gauge.shape[1],
                       posx:posx + downsized_gauge.shape[2]] = opacity * downsized_gauge[1]

            positioned_gauge = torch.zeros_like(img)
            positioned_gauge[:, posy:posy + downsized_gauge.shape[1],
                             posx:posx + downsized_gauge.shape[2]] = downsized_gauge[0]

            img = (1 - mask_gauge) * img + mask_gauge * positioned_gauge
        return img

    def __repr__(self):
        return 'GaugeAugmentation(' + \
            f'{self.numgauges_weights=}, ' + \
            f'{self.scaling_bounds=}, ' + \
            f'{self.opacity_bounds=}, ' + \
            f'{self.gaugechoice_weights=}' + ')'


def augment_idpdataset(dataset, output_size, args, finetune, istrain=True):
    transforms = augment_transforms(output_size, args, dataset.dsbase, finetune=finetune, istrain=istrain)
    print(transforms)

    def augment(d):
        lst = []
        for _ in range(args.n_views if istrain else 1):
            newd = d.copy()  # shallow copy only
            newd['pixel_array'] = transforms(newd['pixel_array'])
            lst.append(newd)
        return lst

    return simonlanger.TDataset(dataset, augment)


# %%


# %%
if __name__ == '__main__' and False:
    # FINDINGS present analysis
    dsbase = IDPDatasetBase()
    df = dsbase.df
    df['findingslen'] = [len(x) for x in df['findings']]
    px.histogram(df, 'findingslen', nbins=50).show()

    df['findingspresent'] = df['findings'].astype(bool)

    regex = r'\s*Untersuchungs-ID: [\w-]+\nUntersuchung:\n\nKlinische Daten:\n\nFragestellung:\n\nVoraufnahmen:\n\nBefund:\n\nBeurteilung:\s*'
    df['findingsnocontent'] = df['findings'].str.match(regex)
    subdf = df[df['findingsnocontent']]['path']
    print(f'no report contents {len(subdf)=}')
    pathsstr = str(list(subdf))

    df['findingsbad'] = (~ df['findingspresent']) | (df['findingsnocontent'])

    fig = px.pie(df, 'bodypart', facet_row='findingsbad', facet_row_spacing=0.2)
    fig.update_traces(textinfo='value+percent')
    fig.update_layout(height=800, width=600)
    fig.show()

    fig = px.histogram(df, 'bodypart', color='findingsbad', text_auto=True)
    fig.data = fig.data[::-1]
    fig.show()

    # df['findingsnumemptycolon'] = df['findings'].str.count(r':\s*\n\s*(\n|[^\n]+:)')
    df['findingsnumemptycolon'] = df['findings'].str.count(r':\s*\n\s*\n')

    df['findingsnumcontentsaftercolon'] = df['findings'].str.count(r': *([\S]+)|(\s*[^\s:]+\n)')

    df['findingsbadnumeric'] = df['findingsbad'].astype(int)

    dimensions = ['bodypart', 'findingsbad', 'findingsnumemptycolon']
    fig = px.parallel_categories(df, dimensions, color='findingsbadnumeric', color_continuous_scale='Jet')
    fig.update_traces(dimensions=[{"categoryorder": "category descending"} for _ in dimensions])
    fig.show('browser')

    dimensions = ['bodypart', 'findingsbad', 'findingsnumcontentsaftercolon']
    fig = px.parallel_categories(df, dimensions, color='findingsbadnumeric', color_continuous_scale='Jet')
    fig.update_traces(dimensions=[{"categoryorder": "category descending"} for _ in dimensions])
    fig.show('browser')

    px.parallel_coordinates(df, ['findingsbadnumeric', 'findingsnumcontentsaftercolon'],
                            color='findingsbadnumeric', color_continuous_scale='Jet').show('browser')

    # for groupkey, group in df.groupby('findingsnumcontentsaftercolon'):
    #     print(groupkey, len(group))
    #     if groupkey in [2,4,5]:
    #         print('\n----------------------\n'.join((group['findings'])))


# %%


# %%
if __name__ == '__main__' and False:
    for path, mappingtype in [(p, p.name[len('BodyPartExamined_mappings_'):][:-len('.json')])
                              for p in
                              sorted(Path('data').glob('BodyPartExamined_mappings_*.json'))
                              ]:
        dsbase = IDPDatasetBase(bodypartexamined_mappingloc=path)
        fig = labelcomparison(dsbase)
        plt.title(f'labelcomparison_{mappingtype}')
        fig.savefig(f'vis/labelcomparison_{mappingtype}.png')

    mappingtype = '_NOMERGE'
    fig = labelcomparison(dsbase, indexcol='dcm_BodyPartExamined_str')
    plt.title(f'labelcomparison_{mappingtype}')
    fig.savefig(f'vis/labelcomparison_{mappingtype}.png')


# %%
if __name__ == '__main__' and False:
    allcols_file = Path('data/cache/dataset_columns.json')
    if not allcols_file.exists():
        dsbase = IDPDatasetBase(required_cols=None)
        allcols_file.write_text(json.dumps(list(dsbase.df.columns), indent=True))

    dsbase = IDPDatasetBase(annotated=True, fracturepseudolabled=True)

    df = dsbase.df
    assert np.all((df['fracture_bestlabel'] == df['fracturenum']) | df['fracturenum'].isna())


# %%
if __name__ == '__main__' and False:
    dss = [
        IDPDatasetFracture(dsbase, 'train', train_include_nopseudolabel=True),
        IDPDatasetFracture(dsbase, 'train', train_include_nopseudolabel=False),
        IDPDatasetFracture(dsbase, 'val'),
        IDPDatasetFracture(dsbase, 'test'),
        IDPDatasetFracture(dsbase, 'train+val', train_include_nopseudolabel=True),
        IDPDatasetFracture(dsbase, 'train+val', train_include_nopseudolabel=False),
        IDPDatasetFracture(dsbase, 'train+val+test', train_include_nopseudolabel=True),
        IDPDatasetFracture(dsbase, 'train+val+test', train_include_nopseudolabel=False),
    ]

    print('\n')
    for ds in dss:
        print(ds)
    
    toconcat = []
    for dstype, ds in zip(['train_all','train_pseudo', 'val', 'test'], dss[:4]):
        df = ds.df.copy()
        df['dstype'] = dstype
        df = df.reset_index()
        toconcat.append(df)

    df = pd.concat(toconcat)

    from IPython.display import display
    
    display(df.pivot_table(values='index',index=['bodypart'], columns=['dstype','fracture_bestlabel'], aggfunc='count', sort=False).fillna(0).astype(int).style.background_gradient(cmap='viridis'))
    
    display(df.pivot_table(values='index',index=['bodypart'], columns=['dstype','foreignmaterial'], aggfunc='count', sort=False).fillna(0).astype(int).style.background_gradient(cmap='viridis'))


# %%
if __name__ == '__main__' and False:
    from train.train_main import prepare_dataloaders

    prep = prepare_dataloaders(types.SimpleNamespace(n_views=1, aug_randomcrop=True, lightly_imagecollate=False, seed=0, normalization_mode=0.99, supervised_proportion=1., overfit=0, prefetch=False, batch_size=256, num_workers=4, perbodypart_training=True))
    display(prep)

# %%
if __name__ == '__main__' and False:
    dsbase = IDPDatasetBase(clean_brightedges=True, clean_rotation=True, annotated=True, size=None)
    # subdf = dsbase.df[(dsbase.df['fracture'] == 'YES') & (dsbase.df['bodypart'] == 'hand')]
    # print('hi')
    # for idx in list(subdf.index)[:100]:
    #     plt.imshow(dsbase[idx]['pixel_array'][0], cmap='bone')
    #     plt.title(str(idx))
    #     plt.show()
    #     #13442

    idx = 13442
    img = dsbase[idx]['pixel_array'][0]
    fig = plt.imshow(img, cmap=plt.cm.bone)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # plt.axis('tight')
    plt.savefig(f'vis/fracture_example.png', dpi=600, pad_inches=0, bbox_inches='tight')
    plt.close()


# %%
if __name__ == '__main__' and False:
    #annotated=True, fracturepseudolabled=True, 
    dsbase = IDPDatasetBase(clean_brightedges=True, clean_rotation=True)
    # curds = IDPDatasetFracture(dsbase, 'train', train_include_nopseudolabel=False)
    curds = IDPDataset(dsbase, 'train')

    # annotated=True, fracturepseudolabled=True, 
    dsbase_noclean = IDPDatasetBase(clean_brightedges=False, clean_rotation=False)
    # curds_noclean = IDPDatasetFracture(dsbase_noclean, 'train', train_include_nopseudolabel=False)
    curds_noclean = IDPDataset(dsbase_noclean, 'train')

    seed = 1337
    torch.manual_seed(seed)
    np.random.seed(seed)

    ds_augmentedstrong = augment_idpdataset(
        curds, (224, 224), types.SimpleNamespace(n_views=1, aug_randomcrop=True, lightly_imagecollate=False), finetune=False, istrain=True)
    ds_augmented = augment_idpdataset(
        curds, (224, 224), types.SimpleNamespace(n_views=1, aug_randomcrop=True, lightly_imagecollate=False), finetune=True, istrain=True)
    ds_augmentedweak = augment_idpdataset(
        curds, (224, 224), types.SimpleNamespace(n_views=1, aug_randomcrop=True, lightly_imagecollate=False), finetune=True, istrain=False)
    # df = dsval_aug.df.reset_index()
    with simonlanger.htmlvis('vis/ds_augmentationvis.html') as builder:
        #6290-3
        for idx in tqdm(range(6290+6,6290+6+1)):
            builder.add_hr()
            builder.add_pltimshow(np.array(curds_noclean[idx]['pixel_array'][0]), cmap='bone')
            builder.add_pltimshow(np.array(curds[idx]['pixel_array'][0]), cmap='bone')
            builder.add_pltimshow(np.array(ds_augmentedstrong[idx][0]['pixel_array'][0]), cmap='bone')
            builder.add_pltimshow(np.array(ds_augmented[idx][0]['pixel_array'][0]), cmap='bone')
            builder.add_pltimshow(np.array(ds_augmented[idx][0]['pixel_array'][0]), cmap='bone')
            builder.add_pltimshow(np.array(ds_augmentedweak[idx][0]['pixel_array'][0]), cmap='bone')
    #     for (bodypart, fracture), subdf in df.groupby(['bodypart', 'fracture']):
    #         scrollid = f'{bodypart=} - {fracture=}'
    #         builder.add_text(scrollid)
    #         builder.add_scrolltarget(scrollid)
    #         for idx in tqdm(subdf.index):
    #             element = dsval_aug[idx]
    #             builder.add_pltimshow(np.array(element[0]['pixel_array'][0]), cmap='bone')
    #         builder.add_hr()

# %%
if __name__ == '__main__' and False:
    dsval_aug = augment_idpdataset(
        IDPDatasetFracture(dsbase, 'val'), (224*2, 224*2), types.SimpleNamespace(n_views=1, aug_randomcrop=True, lightly_imagecollate=False), finetune=False, istrain=False)
    df = dsval_aug.df.reset_index()
    with simonlanger.htmlvis('vis/dsfracture_val_2x.html') as builder:
        for (bodypart, fracture), subdf in df.groupby(['bodypart', 'fracture']):
            scrollid = f'{bodypart=} - {fracture=}'
            builder.add_text(scrollid)
            builder.add_scrolltarget(scrollid)
            for idx in tqdm(subdf.index):
                element = dsval_aug[idx]
                builder.add_pltimshow(np.array(element[0]['pixel_array'][0]), cmap='bone')
            builder.add_hr()

# %%
if __name__ == '__main__' and False:
    dsval_aug = augment_idpdataset(
        IDPDatasetFracture(dsbase, 'train', train_include_nopseudolabel=False), (224*2, 224*2), types.SimpleNamespace(n_views=1, aug_randomcrop=True, lightly_imagecollate=False), finetune=True, istrain=True)
    df = dsval_aug.df.reset_index()
    with simonlanger.htmlvis('vis/dsfracture_train_withpseudolabel_2x.html') as builder:
        for (bodypart, fracture), subdf in df.groupby(['bodypart', 'fracture_bestlabel']):
            scrollid = f'{bodypart=} - {fracture=}'
            builder.add_text(scrollid)
            builder.add_scrolltarget(scrollid)
            import random
            random.seed(0)
            for idx in tqdm(random.sample(list(subdf.index), k=min(50, len(subdf)))):
                element = dsval_aug[idx]
                builder.add_pltimshow(np.array(element[0]['pixel_array'][0]), cmap='bone')
            builder.add_hr()

# %%
if __name__ == '__main__' and False:
    cols = []
    newidx = []
    for ds in dss[1:4]:
        fractures = ds.df[ds.df['fracture_bestlabel'] == 1].groupby('bodypart')['bodypart'].count()
        overall = ds.df.groupby('bodypart')['bodypart'].count()
        cols.append(fractures / overall)
        newidx.append(str(ds))
        cols.append(fractures)
        newidx.append('')
        cols.append(overall)
        newidx.append('')
    df_stats = pd.DataFrame(cols).set_index(pd.Series(newidx)).T
    display(df_stats) # type: ignore


# %%
if __name__ == '__main__' and False:
    for ds in dss[:4]:
        fig = px.parallel_categories(ds.df, ['bodypart', 'fracture_bestlabel',
                                     'annotated'], color='fracture_bestlabel', title=str(ds))
        fig.show()

# %%
if __name__ == '__main__' and False:
    dsbase = IDPDatasetBase(required_cols=None)
    df_annotated = dsbase.df
    df_annotated = df_annotated[df_annotated['annotated']]
    mask_fracture = df_annotated['fracture'].astype(bool)
    mask_foreignmaterial = df_annotated['foreignmaterial'].astype(bool)
    mask_changedbodypart = df_annotated['bodypart'] != df_annotated['bodypart_original']
    masks = [mask_fracture, mask_foreignmaterial, mask_changedbodypart]
    mask_any = functools.reduce(np.logical_or, masks)
    assert len(df_annotated) == mask_any.sum()

    infostr = f'{mask_fracture.sum()=}\n' + \
        f'{mask_foreignmaterial.sum()=}\n' + \
        f'{mask_changedbodypart.sum()=}\n' + \
        f'{mask_any.sum()=}\n'

    print(infostr)
    # Path('vis/annotations_summary.txt').write_text(infostr)

    # df_annotated['fractureint'] = df_annotated['fracture'].factorize(sort=True)[0]
    # df_annotated['foreignmaterialint'] = df_annotated['foreignmaterial'].factorize(sort=True)[0]

    # fig = px.parallel_categories(df_annotated, ['bodypart', 'fracture', 'foreignmaterial'], color='fractureint')
    # fig.write_html('vis/annotations_parallelcategories_byfracture.html', auto_open=True)

    # fig = px.parallel_categories(df_annotated, ['bodypart', 'fracture', 'foreignmaterial'], color='foreignmaterialint')
    # fig.write_html('vis/annotations_parallelcategories_byforeignmaterial.html', auto_open=True)


# %%
if __name__ == '__main__' and False:
    import random
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    for i in random.choices(list(range(len(dstrain))), k=20):
        print(dstrain, i, dstrain.getrow(i)['path'])
        dsres = dstrain[i]
        import matplotlib.pyplot as plt
        plt.imshow(dsres['pixel_array'][0], cmap='bone')
        plt.show()

# %%
if __name__ == '__main__' and False:
    for i, dsres in enumerate(tqdm(dsbase)):
        plt.imshow(dsres['pixel_array'][0], cmap='bone')
        plt.show()
        if i > 10:
            break

# %%
if __name__ == '__main__' and False:
    import random
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    fig = plt.figure(figsize=(12, 12))
    columns = 7
    rows = 7
    dstrain_augmented = augment_idpdataset(
        dstrain, (224, 224), types.SimpleNamespace(n_views=columns * rows, aug_randomcrop=True, lightly_imagecollate=False), finetune=False)
    res = dstrain_augmented[-1]
    for curres, i in zip(res, range(1, columns * rows + 1)):
        img = curres['pixel_array'][0]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='bone')
    plt.show()


# %%

# %%
if __name__ == '__main__' and False:
    import random
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    fig = plt.figure(figsize=(12, 12))
    columns = 7
    rows = 7
    dstrain_augmented = augment_idpdataset(
        dstrain, (224, 224), types.SimpleNamespace(n_views=columns * rows, aug_randomcrop=False, lightly_imagecollate=False), finetune=True)
    res = dstrain_augmented[-1]
    for curres, i in zip(res, range(1, columns * rows + 1)):
        img = curres['pixel_array'][0]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='bone')
    plt.show()


# %%
if __name__ == '__main__':
    dsbase = IDPDatasetBase(required_cols=None)
    dstrain = IDPDataset(dsbase, 'train', seed=0)
    dsval = IDPDataset(dsbase, 'val', seed=0)
    dstest = IDPDataset(dsbase, 'test', seed=0)


# %%
if __name__ == '__main__':
    rows = []

    row = []
    for ds in [dstrain, dsval, dstest]:
        valc = ds.df['dcm_PatientSex'].value_counts()
        row.append(f"{valc['M']}/{valc['F']}")

    rows.append(row)


    # PARSING OF AGE STRING FROM https://github.com/ZviBaratz/dicom_parser/blob/ec61d42824b8da34bc3b0feb212df6d65575e46c/src/dicom_parser/data_elements/age_string.py#L9
    # MIT License
    # adapted by SL

    # N_IN_YEAR is used in order to convert the AgeString value to a
    # standard format of a floating point number representing years.
    N_IN_YEAR = {"Y": 1, "M": 12, "W": 52.1429, "D": 365.2422}

    def parse_value(value: str) -> float:
        """
        Converts an Age String element's representation of age into a *float*
        representing years.

        Parameters
        ----------
        value : str
            Age String value

        Returns
        -------
        float
            Age in years
        """

        if type(value) != str:
            return value
        
        duration = float(value[:-1])
        units = value[-1]
        return duration / N_IN_YEAR[units]

    row = []
    for ds in [dstrain, dsval, dstest]:
        ages = ds.df['dcm_PatientAge'].map(parse_value)
        mean = ages.mean()
        stddev = ages.std()
        iqr = ages.quantile(0.75) - ages.quantile(0.25)
        row.append(f"{mean:.1f} ± {stddev:.1f}, IQR {iqr:.1f}")

    rows.append(row)

    df = pd.DataFrame(rows)
    df = df.rename(index={0:'Sex (M/F)', 1:'Age (Mean ± Standard Deviation, Interquartile Range)'}, 
                   columns={0:'train', 1:'val', 2:'test'})
    df.to_excel('vis/PatientStatistics.xlsx')
    display(df)


# %%
if __name__ == '__main__':
    from scipy.stats import chi2_contingency, ttest_ind
    from scipy.stats.contingency import crosstab

    data = np.array([[15881, 5023], [14924, 4723]])
    # Perform Chi-square test for independence

    chi2, p, dof, expected = chi2_contingency(data)
    print("Chi-square statistic:", chi2.item())
    print("p-value:", p)


    ages_train = dstrain.df['dcm_PatientAge'].map(parse_value).dropna()
    ages_test = dstest.df['dcm_PatientAge'].map(parse_value).dropna()

    print(ttest_ind(ages_train, ages_test, equal_var=False))


  
# %%
